from typing import List, Tuple, Any
from pathlib import Path

from pyaml_env import parse_config
import rosbag
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm

# output folder -> vid1, vid2 ... -> (in_vid) frame_pack_1, frame_pack_2 (60 frames)
# -> (in frame packs) img: 00001.jpg, ann: 00001.lines.txt
# (in txt) 4 rows: x y x y...
# generated: train.txt, val.txt (paths listed inside root folder) + gt masks 
# 3rd degree splines

# vid1, vid2 -> bag_idx

class ImageDataExtractor:
	"""
	"""
	def __init__(self) -> None:
		self.bag = None
		self.cv_bridge = CvBridge()


	def open_rosbag(self, bag_dir: str, bag_name: str) -> rosbag.Bag:
		"""Opens rosbag with the specified path."""
		bag_path = Path(bag_dir) / bag_name
		bag = rosbag.Bag( str(bag_path.resolve()) )
		return bag


	def unpack_ros_topic_data(self, topic_name: str) -> List[Tuple[Any]]:
		"""Unpacks ros data into a list by topic name."""
		data = [
			(topic, msg, t) for (topic, msg, t) in self.bag.read_messages(topics=topic_name)
		]
		return data


	def save_img_from_rosbag(
			self, msg: Any, output_path: Path, is_compressed: bool = True, 
			desired_encoding: str = 'bgr8', is_converted: bool = False, 
			conversion_type: Any = cv2.COLOR_BayerBG2BGR
		) -> None:
		if is_compressed:
			cv_img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding)
		else:
			cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding)
		if is_converted:
			cv_img = cv2.cvtColor(cv_img, conversion_type)
		else:
			pass
		cv2.imwrite( str(output_path.resolve()) , cv_img )


	def init_progressbar(self, description: str, length: int = None) -> Any:
		if length is None:
			length = self.bag.get_message_count()
		pbar = tqdm(total=length)
		pbar.set_description(f"[INFO] {description}: ")
		return pbar


	def process_bag(
		self, 
		bag_dir: str, 
		bag_name: str,
		output_root_dir: str,
		bag_idx: int = 0,
		interval: float = 2.0,
		img_topic_name: str = "/pylon_camera_node/image_raw",
		pack_length: int = 60
	) -> None:
		"""Extracts data from a single rosbag."""

		output_dir = Path(output_root_dir) / f"vid{bag_idx+1}"
		if not output_dir.exists():
			output_dir.mkdir(parents=True)

		train_txt = output_dir / f"train_{bag_idx+1}.txt"
		if not train_txt.exists():
			train_txt.touch()
		
		file_list = output_dir / f"file_list_{bag_idx+1}.yml"
		if not file_list.exists():
			file_list.touch()

		self.bag = self.open_rosbag(bag_dir, bag_name)
		imgs = self.unpack_ros_topic_data(img_topic_name)

		pbar = self.init_progressbar(
			f"Extracting images from bag {bag_idx}", length=len(imgs)
		)

		pack_num = len(imgs) // pack_length
		pack_num_iter = iter([i for i in range(0, pack_num)])

		img_count = 0
		current_t = next(iter(imgs))[2].to_sec()
		print(f"Initial timestamp: {current_t}")
		pack_cnt = next(pack_num_iter)
		for (topic, msg, t) in imgs:
			if img_count >= pack_length:
				pack_cnt = next(pack_num_iter)
				img_count = 0
			timestamp = t.to_sec()
			if timestamp < current_t:
				pass
			else:
				file_name = f"{img_count}".zfill(5)
				pack_dir = output_dir / f"frame_pack_{pack_cnt+1}"
				if not pack_dir.exists():
					pack_dir.mkdir(parents=True)	
				output_path_img = pack_dir / f"{file_name}.jpg"
				self.save_img_from_rosbag(msg, output_path_img)
				output_path_img_rel = str(output_path_img.resolve()).split("/")[-3:]
				output_path_img_rel = "/".join(output_path_img_rel)
				with open(str(train_txt.resolve()), 'a') as f:
					f.write(f"{output_path_img_rel} \n")
				with open(str(file_list.resolve()), 'a') as f:
					f.write(f"- {{url: '/opt/scalabel/local-data/items/{output_path_img_rel}'}} \n")
				current_t += interval
				img_count += 1

			pbar.update()
		pbar.close()

		self.bag.close()

		print(f"[INFO] Bag {bag_idx} extraction was successful!\n")


	def extract(self, 
		bag_dir: str, bag_name_list: List[str], output_root_dir: str,
		interval: float, img_topic_name: str, bag_idx: int = 5
	) -> None:
		"""Extracts data from a list of rosbags."""
		print(bag_name_list)
		for idx, bag_name in enumerate(bag_name_list):
			if idx == bag_idx:
				self.process_bag(
					bag_dir, bag_name, output_root_dir, idx, interval, img_topic_name
				)


if __name__ == '__main__':
	# read parameters
	cfg = parse_config("config/config.yaml")

	bag_dir = cfg["path"]["bag_dir"]
	bag_name_list = [str(path.resolve()).split("/")[-1] for path in Path(bag_dir).glob("*.bag")]
	bag_idx = cfg["path"]["bag_idx"]
	img_topic_name = cfg["ros"]["topic"]["img_topic_name"]
	output_root_dir = cfg["path"]["output_root_dir"]
	interval = cfg["data"]["interval"]

	extractor = ImageDataExtractor()

	extractor.extract(
		bag_dir, bag_name_list, output_root_dir, interval, img_topic_name, bag_idx
	)
