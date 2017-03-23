import os
import tensorflow as tf
import pprint
from g_model import seg_GAN
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("adverarial", False, "Adversarial or normal [50000]")
flags.DEFINE_integer("iterations", 500000, "Epoch to train [50000]")
flags.DEFINE_float("learning_rate", 1e-8, "Learning rate of for SGD [1e-8]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("show_every", 10, "The size of batch images [10]")
flags.DEFINE_integer("save_every", 1000, "save every [1000]")
flags.DEFINE_integer("test_every", 2000, "test every [5000] iterations the subject")
flags.DEFINE_integer("lr_step", 30000, "The step to decrease lr [lr_step]")
flags.DEFINE_integer("sizeCT", 512, "The size of MR patch [512]")
flags.DEFINE_float("wd", 0.0005, "weight decay [0.0005] ")
flags.DEFINE_float("lam_dice", 0.0, "weight dice loss [1.0] ")
flags.DEFINE_float("lam_fcn", 1.0, "weight fcn loss [1.0] ")
flags.DEFINE_float("lam_adv", 1.0, "weight adv loss [1.0] ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dir_patients", "/home/trullro/CT_cleaned/",
 "Directory where the patients are located, the last 4 are used as testing [/home/trullro/CT_cleaned/]")
flags.DEFINE_string("path_patients_h5", "/raid/trullro/unet_h5_2d",
	 "Directory where the h5 files are located ['/raid/trullro/unet_h5_2d']")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #if not os.path.exists(FLAGS.checkpoint_dir):
    #    os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session() as sess:
        gen_model = seg_GAN(sess, batch_size=FLAGS.batch_size, height=FLAGS.sizeCT, width=FLAGS.sizeCT, wd=FLAGS.wd,
                    checkpoint_dir=FLAGS.checkpoint_dir, path_patients_h5=FLAGS.path_patients_h5, learning_rate=FLAGS.learning_rate,
                    lr_step=FLAGS.lr_step,lam_dice=FLAGS.lam_dice, lam_fcn=FLAGS.lam_fcn, lam_adv=FLAGS.lam_adv, adversarial=FLAGS.adverarial)

        if FLAGS.is_train:            
            gen_model.train(FLAGS)
        else:
            print 'Testing mode..'
            if gen_model.load(FLAGS.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                gen_model.sess.run(tf.initialize_all_variables())

          
            start = gen_model.global_step.eval() # get last global_step
            print("test from:", start)
            gen_model.test(FLAGS.dir_patients)


if __name__ == '__main__':
    tf.app.run()
