import numpy as np
from model_GAN import Generator,Discriminator
import tensorflow as tf
import cv2
import logging
tf.logging.set_verbosity(tf.logging.INFO)
def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
def distort(x):
    # heatmap_line_reshape = tf.reshape(x["heatmap_line"], [1, -1, 64, 64, 13])
    # for i in range(1):
    #     heatmap_line_reshape = tf.concat([heatmap_line_reshape, heatmap_line_reshape], 0)
    heatmap_reshape = tf.reshape(x["heatmap"], [1, -1, 64, 64, 68])
    for i in range(1):
        heatmap_reshape = tf.concat([heatmap_reshape, heatmap_reshape], 0)
    return {"heatmap":heatmap_reshape,"heatmap_line":x["heatmap_line"],"heatmap_line_sum":x["heatmap_line_sum"]}
def model_fn(features,labels,mode):
    generator=Generator('G')
    discriminator=Discriminator('D')
    logits_train=generator(features)
    print(logits_train)
    label_tensor = labels
    predictions_dict = { "logits": logits_train}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)
    loss1 = (tf.losses.mean_squared_error(labels=label_tensor["heatmap_line"], predictions=logits_train[0])
            +tf.losses.mean_squared_error(labels=label_tensor["heatmap"], predictions=logits_train[1]))/2-tf.reduce_mean(safe_log(discriminator(generator.image)))
    loss2=(-tf.reduce_mean(safe_log(discriminator(label_tensor["heatmap_line_sum"])))-tf.reduce_mean(safe_log(1-discriminator(generator.image))))/2
    tf.summary.scalar('losss/loss1', loss1)
    tf.summary.scalar('losss/loss2', loss2)
    summary_hook = tf.train.SummarySaverHook(
        100,
        output_dir='train_3D',
        summary_op=tf.summary.merge_all())
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(0.00001, global_step=tf.train.get_global_step(), decay_steps=2000,
                                                   decay_rate=0.95)
        optimizer1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss1, global_step=tf.train.get_global_step(),var_list=generator.variables)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss2,
                                                                           global_step=tf.train.get_global_step(),var_list=discriminator.variables)

        with tf.control_dependencies([optimizer1,optimizer2]):
            train_op=tf.no_op(name='train')
        return tf.estimator.EstimatorSpec(mode=mode, train_op=optimizer1,loss=train_op,
                                          export_outputs={'marks':tf.estimator.export.RegressionOutput(logits_train[1])},training_hooks=[summary_hook])
def _parse_function(record):
    keys_to_features={
        'image':tf.FixedLenFeature([],tf.string),
        'points': tf.FixedLenFeature([136], tf.float32),
        'heatmap':tf.FixedLenFeature([],tf.string),
        'heatmap_line':tf.FixedLenFeature([],tf.string),
    }
    parsed_features=tf.parse_single_example(record,keys_to_features)
    image_decoded=tf.decode_raw(parsed_features['image'], tf.uint8)
    image_reshaped=tf.reshape(image_decoded,[128,128,3])
    image_reshaped=tf.cast(image_reshaped,tf.float32)

    heatmap_line = tf.decode_raw(parsed_features['heatmap_line'], tf.uint8)
    heatmap_line = tf.cast(heatmap_line, tf.float32)
    heatmap_line_reshape = tf.reshape(heatmap_line, [64, 64, 13])
    heatmap_line_sum=tf.reduce_sum(heatmap_line_reshape,axis=2,keepdims=True)
    print(heatmap_line_sum)

    heatmap = tf.decode_raw(parsed_features['heatmap'], tf.uint8)
    heatmap = tf.cast(heatmap, tf.float32)
    heatmap_reshape=tf.reshape(heatmap, [64, 64, 68])
    return {"x":image_reshaped},{"heatmap":heatmap_reshape,"heatmap_line":heatmap_line_reshape,"heatmap_line_sum":heatmap_line_sum}

def input_fn(record_file,batch_size,num_epoches=None,shuffle=True):
    dataset=tf.data.TFRecordDataset(record_file)
    dataset=dataset.map(_parse_function)
    # if shuffle is True:
    #     dataset=dataset.shuffle(buffer_size=20000)
    if batch_size!=1:
        dataset=dataset.batch(batch_size)
    if num_epoches!=1:
        dataset=dataset.repeat(num_epoches)
    iterator=dataset.make_one_shot_iterator()
    fearture,labels=iterator.get_next()
    print(labels)
    labels=distort(labels)
    return fearture,labels

def _train_input_fn():
    return input_fn(record_file="D:/pycharm/dataset/record/data_train_paper_all.record",batch_size=4,num_epoches=10,shuffle=True)
def _predict_input_fn():
    return input_fn(record_file="D:/pycharm/dataset/record/data_train_paper_all.record",batch_size=2,num_epoches=1,shuffle=False)
def main(unused_argv):
    # session_config = tf.ConfigProto(log_device_placement=True)
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # run_config = tf.estimator.RunConfig().replace(session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="train_GAN")
    mode_dict={'train':tf.estimator.ModeKeys.TRAIN,
               'eval':tf.estimator.ModeKeys.EVAL,
               'predict':tf.estimator.ModeKeys.PREDICT}
    mode=mode_dict['train']
    if mode==tf.estimator.ModeKeys.TRAIN:
        estimator.train(input_fn=_train_input_fn,steps=200000)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = estimator.predict(input_fn=_predict_input_fn)
        for _, result in enumerate(predictions):
            imgs=result['logits']
            img = np.array(imgs)[:, :, 2]
            a=np.max(img)
            b=np.array(img).argmax(axis=1)
            c=np.array(b).argmax()
            cv2.imshow("ad",img)
            print(b[c], c)
            cv2.waitKey()
if __name__=="__main__":
    tf.app.run()




