import moviepy.editor as mpe
import numpy as np
import numpy.random
import os.path
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def demo1(sess):
    """Demo based on images dumped during training"""

    # Get images that were dumped during training
    filenames = tf.gfile.ListDirectory(FLAGS.train_dir)
    filenames = sorted(filenames)
    filenames = [os.path.join(FLAGS.train_dir, f) for f in filenames if f[-4:]=='.png']

    assert len(filenames) >= 1

    fps        = 30

    # Create video file from PNGs
    print("正在生成视频中...")
    filename  = os.path.join(FLAGS.train_dir, 'generated_video.mp4')
    clip      = mpe.ImageSequenceClip(filenames, fps=fps)
    clip.write_videofile(filename)
    print("生成完毕，且保存到本地，文件名是：{}。".format(filename))
    
