from tensorflow import keras
import numpy
def one_sided_l2(y_tar, y_pred):
    diff = y_pred - y_tar
    mdiff = diff>0 # mask of positive diff, satisfies constraint
    mdiff32 = keras.backend.cast(mdiff,"float32") # need float32 for product below
    return keras.backend.mean(keras.backend.square(mdiff32*diff), axis=-1)

def my_loss_fn(y_true, y_pred):
    diff = y_true - y_pred
    mDiff = tf.cast(diff>0, "float32")
    squared_difference = tf.square(mDiff*diff)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def xPx_qx(x):
    xPx = keras.backend.transpose(x)@P@x
    qx = keras.backend.transpose(keras.backend.cast(q,"float32"))@x
    return lambda_reg*(0.5*xPx+qx)

def QP_metric(model):
    w = model.layers[0].get_weights()[0]
    return xPx_qx(w)

class ObjHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.objective = []
    def on_epoch_end(self, batch, logs={}):
        self.objective.append(QP_metric(model).numpy()[0][0])

objhist = ObjHistory()

from tensorflow.keras.callbacks import LambdaCallback
print_obj = LambdaCallback(on_epoch_end=lambda batch, logs: print(QP_metric(model)))

P = numpy.asarray([[1.,2.],[0.,4.]])
q = numpy.asarray([[1.],[-1.]])
G = numpy.asarray([[-1.0, 0.0, -1.0, 2.0, 3.0], [ 0.0, -1.0, -3.0, 5.0, 4.0]]).T
h = numpy.asarray([0.0, 0.0, -15.0, 100.0, 80.0]).T
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_dim=2, use_bias=False, kernel_regularizer=xPx_qx))
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
lambda_reg=1e-6
model.compile(optimizer=sgd,              loss=one_sided_l2,              metrics=["mse"])
model.fit(G.reshape(5,2), h.reshape(5,), epochs=1000, callbacks = [print_obj, objhist])