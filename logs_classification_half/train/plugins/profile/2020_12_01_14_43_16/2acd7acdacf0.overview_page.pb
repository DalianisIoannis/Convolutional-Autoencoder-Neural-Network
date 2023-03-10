�	�B���<b@�B���<b@!�B���<b@	 j��? j��?! j��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�B���<b@77�',;S@1�C4��LO@A��5!�1�?I�e�c]|@Y�I�U��?*	G����H[@2F
Iterator::ModeluXᖏ�?!���a�I@)S^+��$�?1��\�C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeato��;���?!�C�T6@)(ђ���?1hp���1@:Preprocessing2U
Iterator::Model::ParallelMapV2�Z� m��?!����&@)�Z� m��?1����&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��e�ik�?!�WX	xE2@)��|�R�?11A�:v�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��B:<��?!��p��X@)��B:<��?1��p��X@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�'�XQ�?!�Qi�qH@)�����z?1<�'d#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=�Еt?!'�Ψ	�@)=�Еt?1'�Ψ	�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@2:=�?!hy���4@)�+���d?1Ń�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 j��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	77�',;S@77�',;S@!77�',;S@      ��!       "	�C4��LO@�C4��LO@!�C4��LO@*      ��!       2	��5!�1�?��5!�1�?!��5!�1�?:	�e�c]|@�e�c]|@!�e�c]|@B      ��!       J	�I�U��?�I�U��?!�I�U��?R      ��!       Z	�I�U��?�I�U��?!�I�U��?JGPUY j��?b �"l
Bgradient_tape/functional_73/conv2d_300/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltert#c2��?!t#c2��?"l
Bgradient_tape/functional_73/conv2d_306/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�}i���?!��P� �?"?
functional_73/conv2d_306/Relu_FusedConv2D^�%�S�?!?���]��?"j
Agradient_tape/functional_73/conv2d_306/Conv2D/Conv2DBackpropInputConv2DBackpropInputY���ӡ�?!v{���?"l
Bgradient_tape/functional_73/conv2d_305/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��l��?!MW1��?"j
Agradient_tape/functional_73/conv2d_305/Conv2D/Conv2DBackpropInputConv2DBackpropInputr�cq��?!�}���?"l
Bgradient_tape/functional_73/conv2d_304/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter
-�^���?!��^뜤�?"l
Bgradient_tape/functional_73/conv2d_303/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�[}�p�?!�. K��?"?
functional_73/conv2d_305/Relu_FusedConv2D�L�FR�?!F	�I�?"?
functional_73/conv2d_304/Relu_FusedConv2DÄ�#&�?!l�{�?Q      Y@Y������@a33333�W@q2u�P*B@y��ԠĖ?"�
both�Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�36.1107% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 