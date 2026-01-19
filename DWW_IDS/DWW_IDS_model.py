import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

class DWW_IDS(tf.keras.Model):
    def __init__(self, n_inp, n_oup=5, his_step=3, n_embding=64,
                 activation='linear', bias=False, en_dropout=0., se_skip=True,):
        super(DWW_IDS, self).__init__()
        self.n_embding = n_embding
        self.n_oup = n_oup

        # Trend-Season Component Decomposition
        self.trand_seasonDecomp = trand_seasonDecomp(kernel_size=2, stride=1)

        # Season vector prediction update
        self.Seasonal_Update = Seasonal_Update()

        # mapping to a high-dimensional space
        if activation == 'linear':
            self.inp_embding = tf.keras.Sequential([
                Embed(n_inp, n_embding // 2, bias=False, proj=activation), layers.ReLU(),
                Embed(n_embding // 2, n_embding, bias=False, proj=activation), layers.ReLU()])

        self.LSTM = layers.LSTM(n_embding, return_sequences=True, return_state=True, dropout=en_dropout)

        # Attention Module
        self.DLAttention = DLAttention(hid_dim=n_embding, init_steps=his_step, skip=se_skip, kernel=2, stride=1)

        # LayerNormalization
        self.LNs = layers.LayerNormalization()

        # Fully connected layer
        self.fc_layers = [Flatten(),
                          layers.Dense(40, activation='relu'),
                          Dropout(0.2),
                          layers.Dense(20, activation='relu'),
                          Dropout(0.2),
                          layers.Dense(n_oup)]  # output

    def call(self, x):
        input_trend, input_season = self.trand_seasonDecomp(x)   # Obtain the trend and season components
        split_season = self.Seasonal_Update(input_season)   # Season components prediction update
        embd_input = input_trend + split_season
        embeddings = self.inp_embding(embd_input)
        en_oup, h_en, c_en = self.LSTM(embeddings)
        att_out = self.DLAttention(en_oup)
        ln_out = self.LNs(att_out)
        x = ln_out
        for layer in self.fc_layers:
            x = layer(x)
        model_output = x

        return model_output

class trand_seasonDecomp(layers.Layer):

    def __init__(self, kernel_size=2, stride=1, **kwargs):
        super(trand_seasonDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_avg = MovingAvg(kernel_size, stride=stride)


    def call(self, x):

        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MovingAvg(layers.Layer):

    def __init__(self, kernel_size=2, stride=1, **kwargs):
        super(MovingAvg, self).__init__(**kwargs)
        self.kernel_size = 2
        self.stride = stride
        self.avg_pool = layers.AveragePooling1D(pool_size=self.kernel_size, strides=self.stride, padding='same')

    def call(self, x):

        x_avg = self.avg_pool(tf.transpose(x, perm=[0, 2, 1]))
        x_avg = tf.transpose(x_avg, perm=[0, 2, 1])
        # Ensure the output length matches the input length
        if x_avg.shape[1] != x.shape[1]:
            x_avg = x_avg[:, :x.shape[1], :]
        return x_avg

class Embed(layers.Layer):
    def __init__(self, in_features, out_features, bias, proj='linear'):
        super(Embed, self).__init__()
        self.proj = proj
        if proj == 'linear':
            self.embed = layers.Dense(out_features, use_bias=bias)

    def call(self, inp):
        if self.proj == 'linear':
            inp = self.embed(inp)
        return inp

class DLAttention(layers.Layer):
    def __init__(self, hid_dim, init_steps, skip=True, kernel=2, stride=1):
        super(DLAttention, self).__init__()
        self.enhance = EnhancedBlock(hid_size=hid_dim, channel=init_steps, skip=skip)
        self.convs = tf.keras.Sequential([
            layers.Conv1D(filters=1, kernel_size=kernel, strides=stride, padding='same', use_bias=False),
            layers.ReLU()
        ])

    def call(self, hid_set):
        reweighted = self.enhance(hid_set)
        align = self.convs(tf.transpose(reweighted, perm=[0, 2, 1]))
        align = tf.transpose(align, perm=[0, 2, 1])
        return align


class EnhancedBlock(layers.Layer):
    def __init__(self, hid_size, channel, skip=True):
        super(EnhancedBlock, self).__init__()
        self.skip = skip
        self.comp = tf.keras.Sequential([
            layers.Dense(hid_size // 2, use_bias=False),
            layers.ReLU(),
            layers.Dense(1, use_bias=False)
        ])
        self.activate = tf.keras.Sequential([
            layers.Dense(channel // 2, use_bias=False),
            layers.ReLU(),
            layers.Dense(channel, use_bias=False),
            layers.Activation('sigmoid')
        ])

    def call(self, inp):

        S = self.comp(inp)
        E = self.activate(tf.transpose(S, perm=[0, 2, 1]))

        out = inp * tf.transpose(E, perm=[0, 2, 1])
        if self.skip:
            out += inp
        return out


class Seasonal_Update(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(Seasonal_Update, self).__init__(**kwargs)
        self.kernel_size = kernel_size

        # 预测和更新模块
        self.predict = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=1, kernel_size=self.kernel_size, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(filters=1, kernel_size=1)
        ])

        self.update = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=1, kernel_size=self.kernel_size, padding='same', activation='relu'),
            tf.keras.layers.Conv1D(filters=1, kernel_size=1)
        ])

    def call(self, inputs):
        x_even = inputs[:, ::2]
        x_odd = inputs[:, 1::2]

        # Prediction step
        predicted = self.predict(x_even)
        updated_c1 = x_odd - predicted

        # Update step
        updated_even = self.update(updated_c1)
        updated_even = x_even + updated_even

        # Vector reconstruction
        reconstructed = tf.concat([updated_even, updated_c1], axis=1)
        reconstructed = tf.sort(reconstructed, axis=1)

        return reconstructed


