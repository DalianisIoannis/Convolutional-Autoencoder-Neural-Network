	-��V�k@-��V�k@!-��V�k@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails--��V�k@A�
�_^@1YO����W@A��j�T�?I!�A	3M@*	V-҅@2Z
#Iterator::Model::ParallelMapV2::ZipX zR&5�?!	�4ZC�O@)�����~�?1�M��C@:Preprocessing2U
Iterator::Model::ParallelMapV2�!�{��?!��Y��4@)�!�{��?1��Y��4@:Preprocessing2F
Iterator::Modeltys�V{�?!�F˥�pB@)���[�?1�����*0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�F���?!I�_�J�-@)YvQ���?1�����*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
~b��?!��rl�1!@)��J��ƪ?1p���a�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��Im �?!��u����?)��Im �?1��u����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx�a��?!pt���?)x�a��?1pt���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6 B\9{�?!���H>�.@))[$�Fo?1�1F5�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	A�
�_^@A�
�_^@!A�
�_^@      ��!       "	YO����W@YO����W@!YO����W@*      ��!       2	��j�T�?��j�T�?!��j�T�?:	!�A	3M@!�A	3M@!!�A	3M@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 