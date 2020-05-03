import rclpy
from rclpy.node import Node
# TODO: cosa cambia tra le 2 immagini?
import sensor_msgs
# from sensor_msgs.msg import Image  # , CompressedImage
from cv_bridge import CvBridge
import PIL
from ros2_segment_topic.models.DeepLabModel import *
import numpy as np


class SegmentAndPublish(Node):

    def __init__(self):
        super().__init__('simple_segmenter')
        self.model = DeepLabModel('/code/data/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz')
        print('model loaded successfully!')

        self.subscription = self.create_subscription(
            sensor_msgs.msg.Image,
            'image/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        self.colormap = self.create_pascal_label_colormap()
        self.publisher_ = self.create_publisher(sensor_msgs.msg.Image, 'image/segmented', 10)

    def listener_callback(self, msg):
        # extract image from msg 
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        im_pil = PIL.Image.fromarray(cv_image)
        resized_im, seg_map = self.model.run(im_pil)
        seg_map = label_to_color_image(seg_map).astype(np.uint8)
        PIL.Image.fromarray(seg_map).save("/code/ros2_ws/data/temp/asd.JPG")
        #seg_map = np.array(resized_im)
        #seg_map = np.array(seg_map)
        # convert image in msg
        #seg_map = np.expand_dims(seg_map,axis=-1)
        print(seg_map.shape);
        # image = self.vis_segmentation_lite(image)
        image = self.bridge.cv2_to_imgmsg(seg_map, "bgr8")
        print("converted")
        self.publisher_.publish(image)
        
    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.

        Returns:
            A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
                ind >>= 3

        return colormap

    def label_to_color_image(self, label):
        """Adds color defined by the dataset colormap to the label.

        Args:
            label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        if np.max(label) >= len(self.colormap):
            raise ValueError('label value too large.')

        return self.colormap[label]

    def vis_segmentation_lite(self, img):
        return self.label_to_color_image(img).astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)

    sap = SegmentAndPublish()

    rclpy.spin(sap)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    sap.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

