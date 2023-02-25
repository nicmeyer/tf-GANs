import tensorflow as tf

from gan import schedule


def repeat_function_call(func, n):
  r = None
  for i in range(n):
    r = func(i)
  return r


class ScheduleTest(tf.test.TestCase):
  def testSchedulerDecaysToMinWhenDecayGreaterThanOne(self):
    init_learning_rate = tf.random.uniform([])
    decay_factor = tf.random.uniform([], 2, 4)
    min_lr = tf.maximum(
      x=self.init_learning_rate - tf.random.uniform([]),
      y=0,
    )
    bound = tf.math.ceil(
      tf.math.log(
        self.init_learning_rate / self.min_lr) / tf.math.log(self.decay_factor))
    bound = tf.cast(bound, tf.int32) + 1
    lr = repeat_function_call(
      func=lambda x: schedule(x, init_learning_rate, decay_factor, min_lr),
      n=bound,
    )
    self.assertEqual(lr, self.min_lr)


if __name__ == '__main__':
  tf.test.main()
