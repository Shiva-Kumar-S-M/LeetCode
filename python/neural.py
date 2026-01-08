import tenserflow as tf
from tenseflow.keras import layers,dense
model = Seqential([
    layers.Dense(64,activation='relu',input_shape=(32,)),
    layers.Dense(10,activation='softmax')
])