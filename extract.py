from typing import List, Tuple, Any
from pathlib import Path

from pyaml_env import parse_config
import rosbag
from cv_bridge import CvBridge
import cv2
from tqdm import tqdm


class ImageDataExtractor:
	"""
	"""
	def __init__(self) -> None:
		self.bag = None


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
			self, msg: Any, output_path: Path, is_compressed: bool = False, 
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
		interval: float = 0.2,
		img_topic_name: str = "/pylon_camera_node/image_raw",
	) -> None:
		"""Extracts data from a single rosbag."""

		output_dir = Path(output_root_dir) / f"{bag_idx}"
		if not output_dir.exists():
			output_dir.mkdir(parents=True)		

		self.bag = self.open_rosbag(bag_dir, bag_name)
		imgs = self.unpack_ros_topic_data(img_topic_name)

		pbar = self.init_progressbar(
			f"Extracting images from bag {bag_idx}", length=len(imgs)
		)

		img_count = 0
		current_t = 0.0
		for (topic, msg, t) in imgs:
			timestamp = t.to_nsec()

			if timestamp < current_t:
				pass
			else:
				file_name = f"{img_count}".zfill(4)
				output_path_img = output_dir / f"{file_name}.png"
				self.save_img_from_rosbag(msg, output_path_img)
				current_t += interval
				img_count += 1

			pbar.update()
		pbar.close()

		self.bag.close()

		print(f"[INFO] Bag {bag_idx} extraction was successful!\n")


	def extract(self, 
		bag_dir: str, bag_name_list: List[str], output_root_dir: str,
		interval: float, img_topic_name: str
	) -> None:
		"""Extracts data from a list of rosbags."""

		for bag_idx, bag_name in enumerate(bag_name_list):
			self.process_bag(
				bag_dir, bag_name, output_root_dir, bag_idx, interval, img_topic_name
			)


if __name__ == '__main__':
	# read parameters
	cfg = parse_config("config/config.yaml")

	bag_dir = cfg["path"]["bag_dir"]
	bag_name_list = [str(path.resolve()).split("/")[-1] for path in Path(bag_dir).glob("*")]
	img_topic_name = cfg["ros"]["topic"]["img_topic_name"]
	output_root_dir = cfg["path"]["output_root_dir"]
	interval = cfg["data"]["interval"]

	extractor = ImageDataExtractor()

	extractor.extract(
		bag_dir, bag_name_list, output_root_dir, img_topic_name
	)