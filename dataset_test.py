
import tensorflow as tf
# Install: pip install tensorflow-datasets
import tensorflow_datasets as tfds
mnist_data = tfds.load("mnist")
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]
assert isinstance(mnist_train, tf.data.Dataset)

ds, info = tfds.load("mnist", split="train", with_info=True)
print(info.splits["train"].num_examples)
print(info.features["label"].num_classes)

ds = ds.batch(128).repeat(10)

for ex in tfds.as_numpy(ds):
	np_image, np_label = ex["image"], ex["label"]
