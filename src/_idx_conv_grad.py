import tensorflow as tf
from tensorflow.python.framework import ops
from _idx_conv import idx_conv_module

@ops.RegisterGradient("IdxConv")
def _idx_conv_grad_cc(op, grad):
    filter_grad = idx_conv_module.idx_conv_grad(grad, op.inputs[0], op.inputs[1],
                                                strides=op.get_attr("strides"), padding=op.get_attr("padding"),
                                                num_bins=op.get_attr("num_bins"), cellsize=op.get_attr("cellsize"),
                                                cells=op.get_attr("cells"), offset=op.get_attr("offset"),
                                                input_size=op.get_attr("input_size"),
                                                anchor_size=op.get_attr("anchor_size"))
    print "grad shape", grad.shape
    print "input0 shape", op.inputs[0].shape
    print "input1 shape", op.inputs[1].shape
    print "input2 shape", op.inputs[2].shape
    print "filter_grad", filter_grad
    #filter_grad = tf.Print(filter_grad, [filter_grad], message="This is filter_grad: ", summarize=100)
    return [None, None, filter_grad]