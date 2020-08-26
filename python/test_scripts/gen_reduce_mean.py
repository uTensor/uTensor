import tensorflow as tf

from model_base import ModelBase


class ReduceMeanModel(tf.keras.Model):
    def __init__(self, axis=None):
        self.axis = axis

    @tf.function
    def call(self, x):
        return tf.math.reduce_mean(x, axis=self.axis)


class QuantReduceMeanModel(ModelBase):
    def __init__(self, model_content):
        self._model_content = model_content


def main():
    pass


if __name__ == "__main__":
    main()
