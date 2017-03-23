import tensorflow as tf
import numpy as np



def loss_dice(logits, labels, num_classes,batch_size_tf):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height].
          The ground truth of your data.
      weights: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    #labels=tf.squeeze(labels)
    with tf.name_scope('loss'):
        #shapelables=labels.get_shape().as_list()
        probs=tf.nn.softmax(logits)        
        y_onehot=tf.one_hot(labels,num_classes,1.0,0.0,axis=3,dtype=tf.float32)
        print 'probs shape ', probs.get_shape()
        print 'y_onehot shape ', y_onehot.get_shape()
        num=tf.reduce_sum(tf.mul(probs,y_onehot), [1,2])
        den1=tf.reduce_sum(tf.mul(probs,probs), [1,2])
        den2=tf.reduce_sum(tf.mul(y_onehot,y_onehot), [1,2])

        dice=2*(num/(den1+den2))
        dice_total=-1*tf.reduce_sum(dice,[1,0])/tf.to_float(batch_size_tf)#divide by batch
        
        #tf.add_to_collection('losses', dice_total)
        loss=dice_total
        #loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def lossfcn(logits, labels, num_classes,batch_size_tf,weights=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      weights: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """

    with tf.name_scope('loss'):
        shapelables=labels.get_shape().as_list()
        #print 'shape labels ',labels.get_shape()
        #print 'shape logits ',logits.get_shape()
        #print 'batch_size_tf ',batch_size_tf
        logits = tf.reshape(logits, [batch_size_tf*shapelables[1]*shapelables[2], num_classes])
        #print 'shape logits reshaped ',logits.get_shape()
        shapelables=labels.get_shape().as_list()
        labels = tf.reshape(labels, [batch_size_tf*shapelables[1]*shapelables[2]])
        labelsonehot=tf.one_hot(labels,num_classes)#Nxnum_classes
        #print labels.get_shape()
        
        if weights is not None:
            labelweights=tf.transpose(tf.matmul(labelsonehot,weights))
            cross_entropy =labelweights*tf.nn.softmax_cross_entropy_with_logits(logits, labelsonehot, name=None)
            
        else:
           
           cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labelsonehot, name=None)
        
        print cross_entropy.get_shape()    
        cross_entropy_mean = tf.reduce_sum(cross_entropy,name='xentropy_mean')/tf.to_float(batch_size_tf)#divide by batch
       
    return cross_entropy_mean