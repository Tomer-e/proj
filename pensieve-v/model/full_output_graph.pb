
g
is_training/Initializer/ConstConst*
value	B
 Z *
_class
loc:@is_training*
dtype0

w
is_training
VariableV2*
dtype0
*
	container *
shape: *
shared_name *
_class
loc:@is_training
f
is_training/AssignIdentityis_training/Initializer/Const*
T0
*
_class
loc:@is_training
R
is_training/readIdentityis_training*
T0
*
_class
loc:@is_training
6
Assign/valueConst*
value	B
 Z*
dtype0

I
AssignIdentityAssign/value*
T0
*
_class
loc:@is_training
8
Assign_1/valueConst*
dtype0
*
value	B
 Z 
M
Assign_1IdentityAssign_1/value*
T0
*
_class
loc:@is_training
:
actor/InputData/XPlaceholder*
dtype0*
shape: 
R
actor/strided_slice/stackConst*!
valueB"        ����*
dtype0
T
actor/strided_slice/stack_1Const*!
valueB"           *
dtype0
T
actor/strided_slice/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_sliceStridedSliceactor/InputData/Xactor/strided_slice/stackactor/strided_slice/stack_1actor/strided_slice/stack_2*
end_mask*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
�
9actor/FullyConnected/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"   �   *)
_class
loc:@actor/FullyConnected/W
�
8actor/FullyConnected/W/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@actor/FullyConnected/W*
dtype0
�
:actor/FullyConnected/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*)
_class
loc:@actor/FullyConnected/W*
dtype0
�
Cactor/FullyConnected/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9actor/FullyConnected/W/Initializer/truncated_normal/shape*
T0*)
_class
loc:@actor/FullyConnected/W*
dtype0*
seed2 *

seed 
�
7actor/FullyConnected/W/Initializer/truncated_normal/mulMulCactor/FullyConnected/W/Initializer/truncated_normal/TruncatedNormal:actor/FullyConnected/W/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@actor/FullyConnected/W
�
3actor/FullyConnected/W/Initializer/truncated_normalAdd7actor/FullyConnected/W/Initializer/truncated_normal/mul8actor/FullyConnected/W/Initializer/truncated_normal/mean*
T0*)
_class
loc:@actor/FullyConnected/W
�
actor/FullyConnected/W
VariableV2*
shared_name *)
_class
loc:@actor/FullyConnected/W*
dtype0*
	container *
shape:	�
�
actor/FullyConnected/W/AssignIdentity3actor/FullyConnected/W/Initializer/truncated_normal*
T0*)
_class
loc:@actor/FullyConnected/W
s
actor/FullyConnected/W/readIdentityactor/FullyConnected/W*
T0*)
_class
loc:@actor/FullyConnected/W
�
(actor/FullyConnected/b/Initializer/ConstConst*
valueB�*    *)
_class
loc:@actor/FullyConnected/b*
dtype0
�
actor/FullyConnected/b
VariableV2*)
_class
loc:@actor/FullyConnected/b*
dtype0*
	container *
shape:�*
shared_name 
�
actor/FullyConnected/b/AssignIdentity(actor/FullyConnected/b/Initializer/Const*
T0*)
_class
loc:@actor/FullyConnected/b
s
actor/FullyConnected/b/readIdentityactor/FullyConnected/b*
T0*)
_class
loc:@actor/FullyConnected/b
�
actor/FullyConnected/MatMulMatMulactor/strided_sliceactor/FullyConnected/W/read*
T0*
transpose_a( *
transpose_b( 
�
actor/FullyConnected/BiasAddBiasAddactor/FullyConnected/MatMulactor/FullyConnected/b/read*
T0*
data_formatNHWC
H
actor/FullyConnected/ReluReluactor/FullyConnected/BiasAdd*
T0
T
actor/strided_slice_1/stackConst*!
valueB"       ����*
dtype0
V
actor/strided_slice_1/stack_1Const*!
valueB"           *
dtype0
V
actor/strided_slice_1/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_slice_1StridedSliceactor/InputData/Xactor/strided_slice_1/stackactor/strided_slice_1/stack_1actor/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
�
;actor/FullyConnected_1/W/Initializer/truncated_normal/shapeConst*
valueB"   �   *+
_class!
loc:@actor/FullyConnected_1/W*
dtype0
�
:actor/FullyConnected_1/W/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@actor/FullyConnected_1/W*
dtype0
�
<actor/FullyConnected_1/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*+
_class!
loc:@actor/FullyConnected_1/W*
dtype0
�
Eactor/FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;actor/FullyConnected_1/W/Initializer/truncated_normal/shape*

seed *
T0*+
_class!
loc:@actor/FullyConnected_1/W*
dtype0*
seed2 
�
9actor/FullyConnected_1/W/Initializer/truncated_normal/mulMulEactor/FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormal<actor/FullyConnected_1/W/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
5actor/FullyConnected_1/W/Initializer/truncated_normalAdd9actor/FullyConnected_1/W/Initializer/truncated_normal/mul:actor/FullyConnected_1/W/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
actor/FullyConnected_1/W
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_1/W*
dtype0*
	container *
shape:	�
�
actor/FullyConnected_1/W/AssignIdentity5actor/FullyConnected_1/W/Initializer/truncated_normal*
T0*+
_class!
loc:@actor/FullyConnected_1/W
y
actor/FullyConnected_1/W/readIdentityactor/FullyConnected_1/W*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
*actor/FullyConnected_1/b/Initializer/ConstConst*
dtype0*
valueB�*    *+
_class!
loc:@actor/FullyConnected_1/b
�
actor/FullyConnected_1/b
VariableV2*+
_class!
loc:@actor/FullyConnected_1/b*
dtype0*
	container *
shape:�*
shared_name 
�
actor/FullyConnected_1/b/AssignIdentity*actor/FullyConnected_1/b/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_1/b
y
actor/FullyConnected_1/b/readIdentityactor/FullyConnected_1/b*
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
actor/FullyConnected_1/MatMulMatMulactor/strided_slice_1actor/FullyConnected_1/W/read*
transpose_a( *
transpose_b( *
T0
�
actor/FullyConnected_1/BiasAddBiasAddactor/FullyConnected_1/MatMulactor/FullyConnected_1/b/read*
T0*
data_formatNHWC
L
actor/FullyConnected_1/ReluReluactor/FullyConnected_1/BiasAdd*
T0
T
actor/strided_slice_2/stackConst*!
valueB"           *
dtype0
V
actor/strided_slice_2/stack_1Const*!
valueB"           *
dtype0
V
actor/strided_slice_2/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_slice_2StridedSliceactor/InputData/Xactor/strided_slice_2/stackactor/strided_slice_2/stack_1actor/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
�
/actor/Conv1D/W/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"         �   *!
_class
loc:@actor/Conv1D/W
}
-actor/Conv1D/W/Initializer/random_uniform/minConst*
dtype0*
valueB
 *qĜ�*!
_class
loc:@actor/Conv1D/W
}
-actor/Conv1D/W/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *qĜ>*!
_class
loc:@actor/Conv1D/W
�
7actor/Conv1D/W/Initializer/random_uniform/RandomUniformRandomUniform/actor/Conv1D/W/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@actor/Conv1D/W*
dtype0*
seed2 
�
-actor/Conv1D/W/Initializer/random_uniform/subSub-actor/Conv1D/W/Initializer/random_uniform/max-actor/Conv1D/W/Initializer/random_uniform/min*
T0*!
_class
loc:@actor/Conv1D/W
�
-actor/Conv1D/W/Initializer/random_uniform/mulMul7actor/Conv1D/W/Initializer/random_uniform/RandomUniform-actor/Conv1D/W/Initializer/random_uniform/sub*
T0*!
_class
loc:@actor/Conv1D/W
�
)actor/Conv1D/W/Initializer/random_uniformAdd-actor/Conv1D/W/Initializer/random_uniform/mul-actor/Conv1D/W/Initializer/random_uniform/min*
T0*!
_class
loc:@actor/Conv1D/W
�
actor/Conv1D/W
VariableV2*
dtype0*
	container *
shape:�*
shared_name *!
_class
loc:@actor/Conv1D/W
x
actor/Conv1D/W/AssignIdentity)actor/Conv1D/W/Initializer/random_uniform*
T0*!
_class
loc:@actor/Conv1D/W
[
actor/Conv1D/W/readIdentityactor/Conv1D/W*
T0*!
_class
loc:@actor/Conv1D/W
u
 actor/Conv1D/b/Initializer/ConstConst*
valueB�*    *!
_class
loc:@actor/Conv1D/b*
dtype0
�
actor/Conv1D/b
VariableV2*!
_class
loc:@actor/Conv1D/b*
dtype0*
	container *
shape:�*
shared_name 
o
actor/Conv1D/b/AssignIdentity actor/Conv1D/b/Initializer/Const*
T0*!
_class
loc:@actor/Conv1D/b
[
actor/Conv1D/b/readIdentityactor/Conv1D/b*
T0*!
_class
loc:@actor/Conv1D/b
E
actor/Conv1D/ExpandDims/dimConst*
value	B :*
dtype0
n
actor/Conv1D/ExpandDims
ExpandDimsactor/strided_slice_2actor/Conv1D/ExpandDims/dim*
T0*

Tdim0
�
actor/Conv1D/Conv2DConv2Dactor/Conv1D/ExpandDimsactor/Conv1D/W/read*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

i
actor/Conv1D/BiasAddBiasAddactor/Conv1D/Conv2Dactor/Conv1D/b/read*
T0*
data_formatNHWC
U
actor/Conv1D/SqueezeSqueezeactor/Conv1D/BiasAdd*
squeeze_dims
*
T0
8
actor/Conv1D/ReluReluactor/Conv1D/Squeeze*
T0
T
actor/strided_slice_3/stackConst*
dtype0*!
valueB"           
V
actor/strided_slice_3/stack_1Const*
dtype0*!
valueB"           
V
actor/strided_slice_3/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_slice_3StridedSliceactor/InputData/Xactor/strided_slice_3/stackactor/strided_slice_3/stack_1actor/strided_slice_3/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask 
�
1actor/Conv1D_1/W/Initializer/random_uniform/shapeConst*%
valueB"         �   *#
_class
loc:@actor/Conv1D_1/W*
dtype0
�
/actor/Conv1D_1/W/Initializer/random_uniform/minConst*
dtype0*
valueB
 *qĜ�*#
_class
loc:@actor/Conv1D_1/W
�
/actor/Conv1D_1/W/Initializer/random_uniform/maxConst*
valueB
 *qĜ>*#
_class
loc:@actor/Conv1D_1/W*
dtype0
�
9actor/Conv1D_1/W/Initializer/random_uniform/RandomUniformRandomUniform1actor/Conv1D_1/W/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@actor/Conv1D_1/W*
dtype0*
seed2 
�
/actor/Conv1D_1/W/Initializer/random_uniform/subSub/actor/Conv1D_1/W/Initializer/random_uniform/max/actor/Conv1D_1/W/Initializer/random_uniform/min*
T0*#
_class
loc:@actor/Conv1D_1/W
�
/actor/Conv1D_1/W/Initializer/random_uniform/mulMul9actor/Conv1D_1/W/Initializer/random_uniform/RandomUniform/actor/Conv1D_1/W/Initializer/random_uniform/sub*
T0*#
_class
loc:@actor/Conv1D_1/W
�
+actor/Conv1D_1/W/Initializer/random_uniformAdd/actor/Conv1D_1/W/Initializer/random_uniform/mul/actor/Conv1D_1/W/Initializer/random_uniform/min*
T0*#
_class
loc:@actor/Conv1D_1/W
�
actor/Conv1D_1/W
VariableV2*
dtype0*
	container *
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_1/W
~
actor/Conv1D_1/W/AssignIdentity+actor/Conv1D_1/W/Initializer/random_uniform*
T0*#
_class
loc:@actor/Conv1D_1/W
a
actor/Conv1D_1/W/readIdentityactor/Conv1D_1/W*
T0*#
_class
loc:@actor/Conv1D_1/W
y
"actor/Conv1D_1/b/Initializer/ConstConst*
dtype0*
valueB�*    *#
_class
loc:@actor/Conv1D_1/b
�
actor/Conv1D_1/b
VariableV2*
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_1/b*
dtype0*
	container 
u
actor/Conv1D_1/b/AssignIdentity"actor/Conv1D_1/b/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_1/b
a
actor/Conv1D_1/b/readIdentityactor/Conv1D_1/b*
T0*#
_class
loc:@actor/Conv1D_1/b
G
actor/Conv1D_1/ExpandDims/dimConst*
value	B :*
dtype0
r
actor/Conv1D_1/ExpandDims
ExpandDimsactor/strided_slice_3actor/Conv1D_1/ExpandDims/dim*

Tdim0*
T0
�
actor/Conv1D_1/Conv2DConv2Dactor/Conv1D_1/ExpandDimsactor/Conv1D_1/W/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
o
actor/Conv1D_1/BiasAddBiasAddactor/Conv1D_1/Conv2Dactor/Conv1D_1/b/read*
T0*
data_formatNHWC
Y
actor/Conv1D_1/SqueezeSqueezeactor/Conv1D_1/BiasAdd*
squeeze_dims
*
T0
<
actor/Conv1D_1/ReluReluactor/Conv1D_1/Squeeze*
T0
T
actor/strided_slice_4/stackConst*
dtype0*!
valueB"           
V
actor/strided_slice_4/stack_1Const*!
valueB"          *
dtype0
V
actor/strided_slice_4/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_slice_4StridedSliceactor/InputData/Xactor/strided_slice_4/stackactor/strided_slice_4/stack_1actor/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask
�
1actor/Conv1D_2/W/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"         �   *#
_class
loc:@actor/Conv1D_2/W
�
/actor/Conv1D_2/W/Initializer/random_uniform/minConst*
valueB
 *���*#
_class
loc:@actor/Conv1D_2/W*
dtype0
�
/actor/Conv1D_2/W/Initializer/random_uniform/maxConst*
valueB
 *��>*#
_class
loc:@actor/Conv1D_2/W*
dtype0
�
9actor/Conv1D_2/W/Initializer/random_uniform/RandomUniformRandomUniform1actor/Conv1D_2/W/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*#
_class
loc:@actor/Conv1D_2/W
�
/actor/Conv1D_2/W/Initializer/random_uniform/subSub/actor/Conv1D_2/W/Initializer/random_uniform/max/actor/Conv1D_2/W/Initializer/random_uniform/min*
T0*#
_class
loc:@actor/Conv1D_2/W
�
/actor/Conv1D_2/W/Initializer/random_uniform/mulMul9actor/Conv1D_2/W/Initializer/random_uniform/RandomUniform/actor/Conv1D_2/W/Initializer/random_uniform/sub*
T0*#
_class
loc:@actor/Conv1D_2/W
�
+actor/Conv1D_2/W/Initializer/random_uniformAdd/actor/Conv1D_2/W/Initializer/random_uniform/mul/actor/Conv1D_2/W/Initializer/random_uniform/min*
T0*#
_class
loc:@actor/Conv1D_2/W
�
actor/Conv1D_2/W
VariableV2*
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_2/W*
dtype0*
	container 
~
actor/Conv1D_2/W/AssignIdentity+actor/Conv1D_2/W/Initializer/random_uniform*
T0*#
_class
loc:@actor/Conv1D_2/W
a
actor/Conv1D_2/W/readIdentityactor/Conv1D_2/W*
T0*#
_class
loc:@actor/Conv1D_2/W
y
"actor/Conv1D_2/b/Initializer/ConstConst*
valueB�*    *#
_class
loc:@actor/Conv1D_2/b*
dtype0
�
actor/Conv1D_2/b
VariableV2*#
_class
loc:@actor/Conv1D_2/b*
dtype0*
	container *
shape:�*
shared_name 
u
actor/Conv1D_2/b/AssignIdentity"actor/Conv1D_2/b/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_2/b
a
actor/Conv1D_2/b/readIdentityactor/Conv1D_2/b*
T0*#
_class
loc:@actor/Conv1D_2/b
G
actor/Conv1D_2/ExpandDims/dimConst*
value	B :*
dtype0
r
actor/Conv1D_2/ExpandDims
ExpandDimsactor/strided_slice_4actor/Conv1D_2/ExpandDims/dim*

Tdim0*
T0
�
actor/Conv1D_2/Conv2DConv2Dactor/Conv1D_2/ExpandDimsactor/Conv1D_2/W/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
	dilations

o
actor/Conv1D_2/BiasAddBiasAddactor/Conv1D_2/Conv2Dactor/Conv1D_2/b/read*
T0*
data_formatNHWC
Y
actor/Conv1D_2/SqueezeSqueezeactor/Conv1D_2/BiasAdd*
squeeze_dims
*
T0
<
actor/Conv1D_2/ReluReluactor/Conv1D_2/Squeeze*
T0
T
actor/strided_slice_5/stackConst*!
valueB"       ����*
dtype0
V
actor/strided_slice_5/stack_1Const*!
valueB"           *
dtype0
V
actor/strided_slice_5/stack_2Const*!
valueB"         *
dtype0
�
actor/strided_slice_5StridedSliceactor/InputData/Xactor/strided_slice_5/stackactor/strided_slice_5/stack_1actor/strided_slice_5/stack_2*
end_mask*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
�
;actor/FullyConnected_2/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"   �   *+
_class!
loc:@actor/FullyConnected_2/W
�
:actor/FullyConnected_2/W/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@actor/FullyConnected_2/W*
dtype0
�
<actor/FullyConnected_2/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*+
_class!
loc:@actor/FullyConnected_2/W*
dtype0
�
Eactor/FullyConnected_2/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;actor/FullyConnected_2/W/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@actor/FullyConnected_2/W*
dtype0*
seed2 *

seed 
�
9actor/FullyConnected_2/W/Initializer/truncated_normal/mulMulEactor/FullyConnected_2/W/Initializer/truncated_normal/TruncatedNormal<actor/FullyConnected_2/W/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
5actor/FullyConnected_2/W/Initializer/truncated_normalAdd9actor/FullyConnected_2/W/Initializer/truncated_normal/mul:actor/FullyConnected_2/W/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
actor/FullyConnected_2/W
VariableV2*+
_class!
loc:@actor/FullyConnected_2/W*
dtype0*
	container *
shape:	�*
shared_name 
�
actor/FullyConnected_2/W/AssignIdentity5actor/FullyConnected_2/W/Initializer/truncated_normal*
T0*+
_class!
loc:@actor/FullyConnected_2/W
y
actor/FullyConnected_2/W/readIdentityactor/FullyConnected_2/W*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
*actor/FullyConnected_2/b/Initializer/ConstConst*
dtype0*
valueB�*    *+
_class!
loc:@actor/FullyConnected_2/b
�
actor/FullyConnected_2/b
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_2/b*
dtype0*
	container *
shape:�
�
actor/FullyConnected_2/b/AssignIdentity*actor/FullyConnected_2/b/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_2/b
y
actor/FullyConnected_2/b/readIdentityactor/FullyConnected_2/b*
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
actor/FullyConnected_2/MatMulMatMulactor/strided_slice_5actor/FullyConnected_2/W/read*
transpose_b( *
T0*
transpose_a( 
�
actor/FullyConnected_2/BiasAddBiasAddactor/FullyConnected_2/MatMulactor/FullyConnected_2/b/read*
T0*
data_formatNHWC
L
actor/FullyConnected_2/ReluReluactor/FullyConnected_2/BiasAdd*
T0
P
actor/Flatten/Reshape/shapeConst*
dtype0*
valueB"�����   
g
actor/Flatten/ReshapeReshapeactor/Conv1D/Reluactor/Flatten/Reshape/shape*
T0*
Tshape0
R
actor/Flatten_1/Reshape/shapeConst*
valueB"�����   *
dtype0
m
actor/Flatten_1/ReshapeReshapeactor/Conv1D_1/Reluactor/Flatten_1/Reshape/shape*
T0*
Tshape0
R
actor/Flatten_2/Reshape/shapeConst*
dtype0*
valueB"�����   
m
actor/Flatten_2/ReshapeReshapeactor/Conv1D_2/Reluactor/Flatten_2/Reshape/shape*
T0*
Tshape0
A
actor/Merge/concat/axisConst*
value	B :*
dtype0
�
actor/Merge/concatConcatV2actor/FullyConnected/Reluactor/FullyConnected_1/Reluactor/Flatten/Reshapeactor/Flatten_1/Reshapeactor/Flatten_2/Reshapeactor/FullyConnected_2/Reluactor/Merge/concat/axis*
N*

Tidx0*
T0
�
;actor/FullyConnected_3/W/Initializer/truncated_normal/shapeConst*
valueB"   �   *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0
�
:actor/FullyConnected_3/W/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0
�
<actor/FullyConnected_3/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*+
_class!
loc:@actor/FullyConnected_3/W*
dtype0
�
Eactor/FullyConnected_3/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;actor/FullyConnected_3/W/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@actor/FullyConnected_3/W*
dtype0*
seed2 *

seed 
�
9actor/FullyConnected_3/W/Initializer/truncated_normal/mulMulEactor/FullyConnected_3/W/Initializer/truncated_normal/TruncatedNormal<actor/FullyConnected_3/W/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
5actor/FullyConnected_3/W/Initializer/truncated_normalAdd9actor/FullyConnected_3/W/Initializer/truncated_normal/mul:actor/FullyConnected_3/W/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
actor/FullyConnected_3/W
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0*
	container *
shape:
��
�
actor/FullyConnected_3/W/AssignIdentity5actor/FullyConnected_3/W/Initializer/truncated_normal*
T0*+
_class!
loc:@actor/FullyConnected_3/W
y
actor/FullyConnected_3/W/readIdentityactor/FullyConnected_3/W*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
*actor/FullyConnected_3/b/Initializer/ConstConst*
dtype0*
valueB�*    *+
_class!
loc:@actor/FullyConnected_3/b
�
actor/FullyConnected_3/b
VariableV2*
shape:�*
shared_name *+
_class!
loc:@actor/FullyConnected_3/b*
dtype0*
	container 
�
actor/FullyConnected_3/b/AssignIdentity*actor/FullyConnected_3/b/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_3/b
y
actor/FullyConnected_3/b/readIdentityactor/FullyConnected_3/b*
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
actor/FullyConnected_3/MatMulMatMulactor/Merge/concatactor/FullyConnected_3/W/read*
transpose_b( *
T0*
transpose_a( 
�
actor/FullyConnected_3/BiasAddBiasAddactor/FullyConnected_3/MatMulactor/FullyConnected_3/b/read*
T0*
data_formatNHWC
L
actor/FullyConnected_3/ReluReluactor/FullyConnected_3/BiasAdd*
T0
�
;actor/FullyConnected_4/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"�      *+
_class!
loc:@actor/FullyConnected_4/W
�
:actor/FullyConnected_4/W/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@actor/FullyConnected_4/W*
dtype0
�
<actor/FullyConnected_4/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*+
_class!
loc:@actor/FullyConnected_4/W*
dtype0
�
Eactor/FullyConnected_4/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;actor/FullyConnected_4/W/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@actor/FullyConnected_4/W*
dtype0*
seed2 *

seed 
�
9actor/FullyConnected_4/W/Initializer/truncated_normal/mulMulEactor/FullyConnected_4/W/Initializer/truncated_normal/TruncatedNormal<actor/FullyConnected_4/W/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
5actor/FullyConnected_4/W/Initializer/truncated_normalAdd9actor/FullyConnected_4/W/Initializer/truncated_normal/mul:actor/FullyConnected_4/W/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
actor/FullyConnected_4/W
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *+
_class!
loc:@actor/FullyConnected_4/W
�
actor/FullyConnected_4/W/AssignIdentity5actor/FullyConnected_4/W/Initializer/truncated_normal*
T0*+
_class!
loc:@actor/FullyConnected_4/W
y
actor/FullyConnected_4/W/readIdentityactor/FullyConnected_4/W*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
*actor/FullyConnected_4/b/Initializer/ConstConst*
valueB*    *+
_class!
loc:@actor/FullyConnected_4/b*
dtype0
�
actor/FullyConnected_4/b
VariableV2*
shape:*
shared_name *+
_class!
loc:@actor/FullyConnected_4/b*
dtype0*
	container 
�
actor/FullyConnected_4/b/AssignIdentity*actor/FullyConnected_4/b/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_4/b
y
actor/FullyConnected_4/b/readIdentityactor/FullyConnected_4/b*
T0*+
_class!
loc:@actor/FullyConnected_4/b
�
actor/FullyConnected_4/MatMulMatMulactor/FullyConnected_3/Reluactor/FullyConnected_4/W/read*
T0*
transpose_a( *
transpose_b( 
�
actor/FullyConnected_4/BiasAddBiasAddactor/FullyConnected_4/MatMulactor/FullyConnected_4/b/read*
data_formatNHWC*
T0
R
actor/FullyConnected_4/SoftmaxSoftmaxactor/FullyConnected_4/BiasAdd*
T0
=
PlaceholderPlaceholder*
dtype0*
shape:	�
;
Placeholder_1Placeholder*
dtype0*
shape:�
?
Placeholder_2Placeholder*
shape:	�*
dtype0
;
Placeholder_3Placeholder*
dtype0*
shape:�
G
Placeholder_4Placeholder*
dtype0*
shape:�
;
Placeholder_5Placeholder*
dtype0*
shape:�
G
Placeholder_6Placeholder*
dtype0*
shape:�
;
Placeholder_7Placeholder*
dtype0*
shape:�
G
Placeholder_8Placeholder*
dtype0*
shape:�
;
Placeholder_9Placeholder*
dtype0*
shape:�
@
Placeholder_10Placeholder*
dtype0*
shape:	�
<
Placeholder_11Placeholder*
shape:�*
dtype0
A
Placeholder_12Placeholder*
dtype0*
shape:
��
<
Placeholder_13Placeholder*
dtype0*
shape:�
@
Placeholder_14Placeholder*
shape:	�*
dtype0
;
Placeholder_15Placeholder*
dtype0*
shape:
U
Assign_2IdentityPlaceholder*
T0*)
_class
loc:@actor/FullyConnected/W
W
Assign_3IdentityPlaceholder_1*
T0*)
_class
loc:@actor/FullyConnected/b
Y
Assign_4IdentityPlaceholder_2*
T0*+
_class!
loc:@actor/FullyConnected_1/W
Y
Assign_5IdentityPlaceholder_3*
T0*+
_class!
loc:@actor/FullyConnected_1/b
O
Assign_6IdentityPlaceholder_4*
T0*!
_class
loc:@actor/Conv1D/W
O
Assign_7IdentityPlaceholder_5*
T0*!
_class
loc:@actor/Conv1D/b
Q
Assign_8IdentityPlaceholder_6*
T0*#
_class
loc:@actor/Conv1D_1/W
Q
Assign_9IdentityPlaceholder_7*
T0*#
_class
loc:@actor/Conv1D_1/b
R
	Assign_10IdentityPlaceholder_8*
T0*#
_class
loc:@actor/Conv1D_2/W
R
	Assign_11IdentityPlaceholder_9*
T0*#
_class
loc:@actor/Conv1D_2/b
[
	Assign_12IdentityPlaceholder_10*
T0*+
_class!
loc:@actor/FullyConnected_2/W
[
	Assign_13IdentityPlaceholder_11*
T0*+
_class!
loc:@actor/FullyConnected_2/b
[
	Assign_14IdentityPlaceholder_12*
T0*+
_class!
loc:@actor/FullyConnected_3/W
[
	Assign_15IdentityPlaceholder_13*
T0*+
_class!
loc:@actor/FullyConnected_3/b
[
	Assign_16IdentityPlaceholder_14*
T0*+
_class!
loc:@actor/FullyConnected_4/W
[
	Assign_17IdentityPlaceholder_15*
T0*+
_class!
loc:@actor/FullyConnected_4/b
7
Placeholder_16Placeholder*
dtype0*
shape: 
7
Placeholder_17Placeholder*
dtype0*
shape: 
C
MulMulactor/FullyConnected_4/SoftmaxPlaceholder_16*
T0
?
Sum/reduction_indicesConst*
dtype0*
value	B :
L
SumSumMulSum/reduction_indices*

Tidx0*
	keep_dims(*
T0

LogLogSum*
T0
#
NegNegPlaceholder_17*
T0

Mul_1MulLogNeg*
T0
:
ConstConst*
valueB"       *
dtype0
@
Sum_1SumMul_1Const*
T0*

Tidx0*
	keep_dims( 
2
add/yConst*
dtype0*
valueB
 *�7�5
:
addAddactor/FullyConnected_4/Softmaxadd/y*
T0

Log_1Logadd*
T0
<
Mul_2Mulactor/FullyConnected_4/SoftmaxLog_1*
T0
<
Const_1Const*
dtype0*
valueB"       
B
Sum_2SumMul_2Const_1*
T0*

Tidx0*
	keep_dims( 
2
mul/xConst*
valueB
 *
ף<*
dtype0
!
mulMulmul/xSum_2*
T0
!
add_1AddSum_1mul*
T0
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
dtype0*
valueB
 *  �?
S
gradients/FillFillgradients/Shapegradients/Const*
T0*

index_type0
C
gradients/add_1_grad/ShapeConst*
dtype0*
valueB 
E
gradients/add_1_grad/Shape_1Const*
dtype0*
valueB 
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0
�
gradients/add_1_grad/SumSumgradients/Fill*gradients/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sumgradients/Fill,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0
W
"gradients/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB"      
�
gradients/Sum_1_grad/ReshapeReshapegradients/add_1_grad/Reshape"gradients/Sum_1_grad/Reshape/shape*
T0*
Tshape0
C
gradients/Sum_1_grad/ShapeShapeMul_1*
T0*
out_type0
v
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/Shape*
T0*

Tmultiples0
A
gradients/mul_grad/ShapeConst*
valueB *
dtype0
C
gradients/mul_grad/Shape_1Const*
valueB *
dtype0
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
M
gradients/mul_grad/mulMulgradients/add_1_grad/Reshape_1Sum_2*
T0
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
O
gradients/mul_grad/mul_1Mulmul/xgradients/add_1_grad/Reshape_1*
T0
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
A
gradients/Mul_1_grad/ShapeShapeLog*
T0*
out_type0
C
gradients/Mul_1_grad/Shape_1ShapeNeg*
T0*
out_type0
�
*gradients/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_1_grad/Shapegradients/Mul_1_grad/Shape_1*
T0
H
gradients/Mul_1_grad/mulMulgradients/Sum_1_grad/TileNeg*
T0
�
gradients/Mul_1_grad/SumSumgradients/Mul_1_grad/mul*gradients/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/Mul_1_grad/ReshapeReshapegradients/Mul_1_grad/Sumgradients/Mul_1_grad/Shape*
T0*
Tshape0
J
gradients/Mul_1_grad/mul_1MulLoggradients/Sum_1_grad/Tile*
T0
�
gradients/Mul_1_grad/Sum_1Sumgradients/Mul_1_grad/mul_1,gradients/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients/Mul_1_grad/Reshape_1Reshapegradients/Mul_1_grad/Sum_1gradients/Mul_1_grad/Shape_1*
T0*
Tshape0
W
"gradients/Sum_2_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients/Sum_2_grad/ReshapeReshapegradients/mul_grad/Reshape_1"gradients/Sum_2_grad/Reshape/shape*
T0*
Tshape0
C
gradients/Sum_2_grad/ShapeShapeMul_2*
T0*
out_type0
v
gradients/Sum_2_grad/TileTilegradients/Sum_2_grad/Reshapegradients/Sum_2_grad/Shape*
T0*

Tmultiples0
X
gradients/Log_grad/Reciprocal
ReciprocalSum^gradients/Mul_1_grad/Reshape*
T0
c
gradients/Log_grad/mulMulgradients/Mul_1_grad/Reshapegradients/Log_grad/Reciprocal*
T0
\
gradients/Mul_2_grad/ShapeShapeactor/FullyConnected_4/Softmax*
T0*
out_type0
E
gradients/Mul_2_grad/Shape_1ShapeLog_1*
T0*
out_type0
�
*gradients/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_2_grad/Shapegradients/Mul_2_grad/Shape_1*
T0
J
gradients/Mul_2_grad/mulMulgradients/Sum_2_grad/TileLog_1*
T0
�
gradients/Mul_2_grad/SumSumgradients/Mul_2_grad/mul*gradients/Mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients/Mul_2_grad/ReshapeReshapegradients/Mul_2_grad/Sumgradients/Mul_2_grad/Shape*
T0*
Tshape0
e
gradients/Mul_2_grad/mul_1Mulactor/FullyConnected_4/Softmaxgradients/Sum_2_grad/Tile*
T0
�
gradients/Mul_2_grad/Sum_1Sumgradients/Mul_2_grad/mul_1,gradients/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/Mul_2_grad/Reshape_1Reshapegradients/Mul_2_grad/Sum_1gradients/Mul_2_grad/Shape_1*
T0*
Tshape0
?
gradients/Sum_grad/ShapeShapeMul*
T0*
out_type0
A
gradients/Sum_grad/SizeConst*
value	B :*
dtype0
V
gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
T0
\
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0
C
gradients/Sum_grad/Shape_1Const*
valueB *
dtype0
H
gradients/Sum_grad/range/startConst*
value	B : *
dtype0
H
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0
G
gradients/Sum_grad/Fill/valueConst*
dtype0*
value	B :
u
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*

index_type0
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*
N
F
gradients/Sum_grad/Maximum/yConst*
value	B :*
dtype0
n
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0
f
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0
v
gradients/Sum_grad/ReshapeReshapegradients/Log_grad/mul gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0
\
gradients/Log_1_grad/Reciprocal
Reciprocaladd^gradients/Mul_2_grad/Reshape_1*
T0
i
gradients/Log_1_grad/mulMulgradients/Mul_2_grad/Reshape_1gradients/Log_1_grad/Reciprocal*
T0
Z
gradients/Mul_grad/ShapeShapeactor/FullyConnected_4/Softmax*
T0*
out_type0
L
gradients/Mul_grad/Shape_1ShapePlaceholder_16*
T0*
out_type0
�
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0
O
gradients/Mul_grad/mulMulgradients/Sum_grad/TilePlaceholder_16*
T0
�
gradients/Mul_grad/SumSumgradients/Mul_grad/mul(gradients/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0
a
gradients/Mul_grad/mul_1Mulactor/FullyConnected_4/Softmaxgradients/Sum_grad/Tile*
T0
�
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
t
gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0
Z
gradients/add_grad/ShapeShapeactor/FullyConnected_4/Softmax*
T0*
out_type0
C
gradients/add_grad/Shape_1Const*
dtype0*
valueB 
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
�
gradients/add_grad/SumSumgradients/Log_1_grad/mul(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
�
gradients/AddNAddNgradients/Mul_2_grad/Reshapegradients/Mul_grad/Reshapegradients/add_grad/Reshape*
N*
T0*/
_class%
#!loc:@gradients/Mul_2_grad/Reshape
q
1gradients/actor/FullyConnected_4/Softmax_grad/mulMulgradients/AddNactor/FullyConnected_4/Softmax*
T0
q
Cgradients/actor/FullyConnected_4/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB:
�
1gradients/actor/FullyConnected_4/Softmax_grad/SumSum1gradients/actor/FullyConnected_4/Softmax_grad/mulCgradients/actor/FullyConnected_4/Softmax_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
p
;gradients/actor/FullyConnected_4/Softmax_grad/Reshape/shapeConst*
valueB"����   *
dtype0
�
5gradients/actor/FullyConnected_4/Softmax_grad/ReshapeReshape1gradients/actor/FullyConnected_4/Softmax_grad/Sum;gradients/actor/FullyConnected_4/Softmax_grad/Reshape/shape*
T0*
Tshape0
�
1gradients/actor/FullyConnected_4/Softmax_grad/subSubgradients/AddN5gradients/actor/FullyConnected_4/Softmax_grad/Reshape*
T0
�
3gradients/actor/FullyConnected_4/Softmax_grad/mul_1Mul1gradients/actor/FullyConnected_4/Softmax_grad/subactor/FullyConnected_4/Softmax*
T0
�
9gradients/actor/FullyConnected_4/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/actor/FullyConnected_4/Softmax_grad/mul_1*
data_formatNHWC*
T0
�
3gradients/actor/FullyConnected_4/MatMul_grad/MatMulMatMul3gradients/actor/FullyConnected_4/Softmax_grad/mul_1actor/FullyConnected_4/W/read*
T0*
transpose_a( *
transpose_b(
�
5gradients/actor/FullyConnected_4/MatMul_grad/MatMul_1MatMulactor/FullyConnected_3/Relu3gradients/actor/FullyConnected_4/Softmax_grad/mul_1*
transpose_a(*
transpose_b( *
T0
�
3gradients/actor/FullyConnected_3/Relu_grad/ReluGradReluGrad3gradients/actor/FullyConnected_4/MatMul_grad/MatMulactor/FullyConnected_3/Relu*
T0
�
9gradients/actor/FullyConnected_3/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/actor/FullyConnected_3/Relu_grad/ReluGrad*
data_formatNHWC*
T0
�
3gradients/actor/FullyConnected_3/MatMul_grad/MatMulMatMul3gradients/actor/FullyConnected_3/Relu_grad/ReluGradactor/FullyConnected_3/W/read*
transpose_a( *
transpose_b(*
T0
�
5gradients/actor/FullyConnected_3/MatMul_grad/MatMul_1MatMulactor/Merge/concat3gradients/actor/FullyConnected_3/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
P
&gradients/actor/Merge/concat_grad/RankConst*
dtype0*
value	B :
{
%gradients/actor/Merge/concat_grad/modFloorModactor/Merge/concat/axis&gradients/actor/Merge/concat_grad/Rank*
T0
d
'gradients/actor/Merge/concat_grad/ShapeShapeactor/FullyConnected/Relu*
T0*
out_type0
�
(gradients/actor/Merge/concat_grad/ShapeNShapeNactor/FullyConnected/Reluactor/FullyConnected_1/Reluactor/Flatten/Reshapeactor/Flatten_1/Reshapeactor/Flatten_2/Reshapeactor/FullyConnected_2/Relu*
N*
T0*
out_type0
�
.gradients/actor/Merge/concat_grad/ConcatOffsetConcatOffset%gradients/actor/Merge/concat_grad/mod(gradients/actor/Merge/concat_grad/ShapeN*gradients/actor/Merge/concat_grad/ShapeN:1*gradients/actor/Merge/concat_grad/ShapeN:2*gradients/actor/Merge/concat_grad/ShapeN:3*gradients/actor/Merge/concat_grad/ShapeN:4*gradients/actor/Merge/concat_grad/ShapeN:5*
N
�
'gradients/actor/Merge/concat_grad/SliceSlice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul.gradients/actor/Merge/concat_grad/ConcatOffset(gradients/actor/Merge/concat_grad/ShapeN*
T0*
Index0
�
)gradients/actor/Merge/concat_grad/Slice_1Slice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul0gradients/actor/Merge/concat_grad/ConcatOffset:1*gradients/actor/Merge/concat_grad/ShapeN:1*
T0*
Index0
�
)gradients/actor/Merge/concat_grad/Slice_2Slice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul0gradients/actor/Merge/concat_grad/ConcatOffset:2*gradients/actor/Merge/concat_grad/ShapeN:2*
T0*
Index0
�
)gradients/actor/Merge/concat_grad/Slice_3Slice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul0gradients/actor/Merge/concat_grad/ConcatOffset:3*gradients/actor/Merge/concat_grad/ShapeN:3*
T0*
Index0
�
)gradients/actor/Merge/concat_grad/Slice_4Slice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul0gradients/actor/Merge/concat_grad/ConcatOffset:4*gradients/actor/Merge/concat_grad/ShapeN:4*
T0*
Index0
�
)gradients/actor/Merge/concat_grad/Slice_5Slice3gradients/actor/FullyConnected_3/MatMul_grad/MatMul0gradients/actor/Merge/concat_grad/ConcatOffset:5*gradients/actor/Merge/concat_grad/ShapeN:5*
T0*
Index0
�
1gradients/actor/FullyConnected/Relu_grad/ReluGradReluGrad'gradients/actor/Merge/concat_grad/Sliceactor/FullyConnected/Relu*
T0
�
3gradients/actor/FullyConnected_1/Relu_grad/ReluGradReluGrad)gradients/actor/Merge/concat_grad/Slice_1actor/FullyConnected_1/Relu*
T0
_
*gradients/actor/Flatten/Reshape_grad/ShapeShapeactor/Conv1D/Relu*
T0*
out_type0
�
,gradients/actor/Flatten/Reshape_grad/ReshapeReshape)gradients/actor/Merge/concat_grad/Slice_2*gradients/actor/Flatten/Reshape_grad/Shape*
T0*
Tshape0
c
,gradients/actor/Flatten_1/Reshape_grad/ShapeShapeactor/Conv1D_1/Relu*
T0*
out_type0
�
.gradients/actor/Flatten_1/Reshape_grad/ReshapeReshape)gradients/actor/Merge/concat_grad/Slice_3,gradients/actor/Flatten_1/Reshape_grad/Shape*
T0*
Tshape0
c
,gradients/actor/Flatten_2/Reshape_grad/ShapeShapeactor/Conv1D_2/Relu*
T0*
out_type0
�
.gradients/actor/Flatten_2/Reshape_grad/ReshapeReshape)gradients/actor/Merge/concat_grad/Slice_4,gradients/actor/Flatten_2/Reshape_grad/Shape*
T0*
Tshape0
�
3gradients/actor/FullyConnected_2/Relu_grad/ReluGradReluGrad)gradients/actor/Merge/concat_grad/Slice_5actor/FullyConnected_2/Relu*
T0
�
7gradients/actor/FullyConnected/BiasAdd_grad/BiasAddGradBiasAddGrad1gradients/actor/FullyConnected/Relu_grad/ReluGrad*
data_formatNHWC*
T0
�
9gradients/actor/FullyConnected_1/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/actor/FullyConnected_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC

)gradients/actor/Conv1D/Relu_grad/ReluGradReluGrad,gradients/actor/Flatten/Reshape_grad/Reshapeactor/Conv1D/Relu*
T0
�
+gradients/actor/Conv1D_1/Relu_grad/ReluGradReluGrad.gradients/actor/Flatten_1/Reshape_grad/Reshapeactor/Conv1D_1/Relu*
T0
�
+gradients/actor/Conv1D_2/Relu_grad/ReluGradReluGrad.gradients/actor/Flatten_2/Reshape_grad/Reshapeactor/Conv1D_2/Relu*
T0
�
9gradients/actor/FullyConnected_2/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/actor/FullyConnected_2/Relu_grad/ReluGrad*
data_formatNHWC*
T0
�
1gradients/actor/FullyConnected/MatMul_grad/MatMulMatMul1gradients/actor/FullyConnected/Relu_grad/ReluGradactor/FullyConnected/W/read*
T0*
transpose_a( *
transpose_b(
�
3gradients/actor/FullyConnected/MatMul_grad/MatMul_1MatMulactor/strided_slice1gradients/actor/FullyConnected/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
�
3gradients/actor/FullyConnected_1/MatMul_grad/MatMulMatMul3gradients/actor/FullyConnected_1/Relu_grad/ReluGradactor/FullyConnected_1/W/read*
transpose_a( *
transpose_b(*
T0
�
5gradients/actor/FullyConnected_1/MatMul_grad/MatMul_1MatMulactor/strided_slice_13gradients/actor/FullyConnected_1/Relu_grad/ReluGrad*
transpose_b( *
T0*
transpose_a(
a
)gradients/actor/Conv1D/Squeeze_grad/ShapeShapeactor/Conv1D/BiasAdd*
T0*
out_type0
�
+gradients/actor/Conv1D/Squeeze_grad/ReshapeReshape)gradients/actor/Conv1D/Relu_grad/ReluGrad)gradients/actor/Conv1D/Squeeze_grad/Shape*
T0*
Tshape0
e
+gradients/actor/Conv1D_1/Squeeze_grad/ShapeShapeactor/Conv1D_1/BiasAdd*
T0*
out_type0
�
-gradients/actor/Conv1D_1/Squeeze_grad/ReshapeReshape+gradients/actor/Conv1D_1/Relu_grad/ReluGrad+gradients/actor/Conv1D_1/Squeeze_grad/Shape*
T0*
Tshape0
e
+gradients/actor/Conv1D_2/Squeeze_grad/ShapeShapeactor/Conv1D_2/BiasAdd*
T0*
out_type0
�
-gradients/actor/Conv1D_2/Squeeze_grad/ReshapeReshape+gradients/actor/Conv1D_2/Relu_grad/ReluGrad+gradients/actor/Conv1D_2/Squeeze_grad/Shape*
T0*
Tshape0
�
3gradients/actor/FullyConnected_2/MatMul_grad/MatMulMatMul3gradients/actor/FullyConnected_2/Relu_grad/ReluGradactor/FullyConnected_2/W/read*
T0*
transpose_a( *
transpose_b(
�
5gradients/actor/FullyConnected_2/MatMul_grad/MatMul_1MatMulactor/strided_slice_53gradients/actor/FullyConnected_2/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
�
/gradients/actor/Conv1D/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/actor/Conv1D/Squeeze_grad/Reshape*
data_formatNHWC*
T0
�
1gradients/actor/Conv1D_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/actor/Conv1D_1/Squeeze_grad/Reshape*
T0*
data_formatNHWC
�
1gradients/actor/Conv1D_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/actor/Conv1D_2/Squeeze_grad/Reshape*
T0*
data_formatNHWC
c
(gradients/actor/Conv1D/Conv2D_grad/ShapeShapeactor/Conv1D/ExpandDims*
T0*
out_type0
�
6gradients/actor/Conv1D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput(gradients/actor/Conv1D/Conv2D_grad/Shapeactor/Conv1D/W/read+gradients/actor/Conv1D/Squeeze_grad/Reshape*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
	dilations

g
*gradients/actor/Conv1D/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0
�
7gradients/actor/Conv1D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteractor/Conv1D/ExpandDims*gradients/actor/Conv1D/Conv2D_grad/Shape_1+gradients/actor/Conv1D/Squeeze_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
g
*gradients/actor/Conv1D_1/Conv2D_grad/ShapeShapeactor/Conv1D_1/ExpandDims*
T0*
out_type0
�
8gradients/actor/Conv1D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/actor/Conv1D_1/Conv2D_grad/Shapeactor/Conv1D_1/W/read-gradients/actor/Conv1D_1/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
i
,gradients/actor/Conv1D_1/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0
�
9gradients/actor/Conv1D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteractor/Conv1D_1/ExpandDims,gradients/actor/Conv1D_1/Conv2D_grad/Shape_1-gradients/actor/Conv1D_1/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
g
*gradients/actor/Conv1D_2/Conv2D_grad/ShapeShapeactor/Conv1D_2/ExpandDims*
T0*
out_type0
�
8gradients/actor/Conv1D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput*gradients/actor/Conv1D_2/Conv2D_grad/Shapeactor/Conv1D_2/W/read-gradients/actor/Conv1D_2/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
i
,gradients/actor/Conv1D_2/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0
�
9gradients/actor/Conv1D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilteractor/Conv1D_2/ExpandDims,gradients/actor/Conv1D_2/Conv2D_grad/Shape_1-gradients/actor/Conv1D_2/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
�
0actor/FullyConnected/W/RMSProp/Initializer/ConstConst*
dtype0*
valueB	�*  �?*)
_class
loc:@actor/FullyConnected/W
�
actor/FullyConnected/W/RMSProp
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *)
_class
loc:@actor/FullyConnected/W
�
%actor/FullyConnected/W/RMSProp/AssignIdentity0actor/FullyConnected/W/RMSProp/Initializer/Const*
T0*)
_class
loc:@actor/FullyConnected/W
�
#actor/FullyConnected/W/RMSProp/readIdentityactor/FullyConnected/W/RMSProp*
T0*)
_class
loc:@actor/FullyConnected/W
�
2actor/FullyConnected/W/RMSProp_1/Initializer/ConstConst*
valueB	�*    *)
_class
loc:@actor/FullyConnected/W*
dtype0
�
 actor/FullyConnected/W/RMSProp_1
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *)
_class
loc:@actor/FullyConnected/W
�
'actor/FullyConnected/W/RMSProp_1/AssignIdentity2actor/FullyConnected/W/RMSProp_1/Initializer/Const*
T0*)
_class
loc:@actor/FullyConnected/W
�
%actor/FullyConnected/W/RMSProp_1/readIdentity actor/FullyConnected/W/RMSProp_1*
T0*)
_class
loc:@actor/FullyConnected/W
�
0actor/FullyConnected/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*)
_class
loc:@actor/FullyConnected/b*
dtype0
�
actor/FullyConnected/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *)
_class
loc:@actor/FullyConnected/b
�
%actor/FullyConnected/b/RMSProp/AssignIdentity0actor/FullyConnected/b/RMSProp/Initializer/Const*
T0*)
_class
loc:@actor/FullyConnected/b
�
#actor/FullyConnected/b/RMSProp/readIdentityactor/FullyConnected/b/RMSProp*
T0*)
_class
loc:@actor/FullyConnected/b
�
2actor/FullyConnected/b/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB�*    *)
_class
loc:@actor/FullyConnected/b
�
 actor/FullyConnected/b/RMSProp_1
VariableV2*
shared_name *)
_class
loc:@actor/FullyConnected/b*
dtype0*
	container *
shape:�
�
'actor/FullyConnected/b/RMSProp_1/AssignIdentity2actor/FullyConnected/b/RMSProp_1/Initializer/Const*
T0*)
_class
loc:@actor/FullyConnected/b
�
%actor/FullyConnected/b/RMSProp_1/readIdentity actor/FullyConnected/b/RMSProp_1*
T0*)
_class
loc:@actor/FullyConnected/b
�
2actor/FullyConnected_1/W/RMSProp/Initializer/ConstConst*
valueB	�*  �?*+
_class!
loc:@actor/FullyConnected_1/W*
dtype0
�
 actor/FullyConnected_1/W/RMSProp
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *+
_class!
loc:@actor/FullyConnected_1/W
�
'actor/FullyConnected_1/W/RMSProp/AssignIdentity2actor/FullyConnected_1/W/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
%actor/FullyConnected_1/W/RMSProp/readIdentity actor/FullyConnected_1/W/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
4actor/FullyConnected_1/W/RMSProp_1/Initializer/ConstConst*
valueB	�*    *+
_class!
loc:@actor/FullyConnected_1/W*
dtype0
�
"actor/FullyConnected_1/W/RMSProp_1
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_1/W*
dtype0*
	container *
shape:	�
�
)actor/FullyConnected_1/W/RMSProp_1/AssignIdentity4actor/FullyConnected_1/W/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
'actor/FullyConnected_1/W/RMSProp_1/readIdentity"actor/FullyConnected_1/W/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
2actor/FullyConnected_1/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*+
_class!
loc:@actor/FullyConnected_1/b
�
 actor/FullyConnected_1/b/RMSProp
VariableV2*+
_class!
loc:@actor/FullyConnected_1/b*
dtype0*
	container *
shape:�*
shared_name 
�
'actor/FullyConnected_1/b/RMSProp/AssignIdentity2actor/FullyConnected_1/b/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
%actor/FullyConnected_1/b/RMSProp/readIdentity actor/FullyConnected_1/b/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
4actor/FullyConnected_1/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *+
_class!
loc:@actor/FullyConnected_1/b*
dtype0
�
"actor/FullyConnected_1/b/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *+
_class!
loc:@actor/FullyConnected_1/b
�
)actor/FullyConnected_1/b/RMSProp_1/AssignIdentity4actor/FullyConnected_1/b/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
'actor/FullyConnected_1/b/RMSProp_1/readIdentity"actor/FullyConnected_1/b/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
(actor/Conv1D/W/RMSProp/Initializer/ConstConst*&
valueB�*  �?*!
_class
loc:@actor/Conv1D/W*
dtype0
�
actor/Conv1D/W/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *!
_class
loc:@actor/Conv1D/W

actor/Conv1D/W/RMSProp/AssignIdentity(actor/Conv1D/W/RMSProp/Initializer/Const*
T0*!
_class
loc:@actor/Conv1D/W
k
actor/Conv1D/W/RMSProp/readIdentityactor/Conv1D/W/RMSProp*
T0*!
_class
loc:@actor/Conv1D/W
�
*actor/Conv1D/W/RMSProp_1/Initializer/ConstConst*
dtype0*&
valueB�*    *!
_class
loc:@actor/Conv1D/W
�
actor/Conv1D/W/RMSProp_1
VariableV2*
shape:�*
shared_name *!
_class
loc:@actor/Conv1D/W*
dtype0*
	container 
�
actor/Conv1D/W/RMSProp_1/AssignIdentity*actor/Conv1D/W/RMSProp_1/Initializer/Const*
T0*!
_class
loc:@actor/Conv1D/W
o
actor/Conv1D/W/RMSProp_1/readIdentityactor/Conv1D/W/RMSProp_1*
T0*!
_class
loc:@actor/Conv1D/W
}
(actor/Conv1D/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*!
_class
loc:@actor/Conv1D/b
�
actor/Conv1D/b/RMSProp
VariableV2*
shared_name *!
_class
loc:@actor/Conv1D/b*
dtype0*
	container *
shape:�

actor/Conv1D/b/RMSProp/AssignIdentity(actor/Conv1D/b/RMSProp/Initializer/Const*
T0*!
_class
loc:@actor/Conv1D/b
k
actor/Conv1D/b/RMSProp/readIdentityactor/Conv1D/b/RMSProp*
T0*!
_class
loc:@actor/Conv1D/b

*actor/Conv1D/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *!
_class
loc:@actor/Conv1D/b*
dtype0
�
actor/Conv1D/b/RMSProp_1
VariableV2*
shared_name *!
_class
loc:@actor/Conv1D/b*
dtype0*
	container *
shape:�
�
actor/Conv1D/b/RMSProp_1/AssignIdentity*actor/Conv1D/b/RMSProp_1/Initializer/Const*
T0*!
_class
loc:@actor/Conv1D/b
o
actor/Conv1D/b/RMSProp_1/readIdentityactor/Conv1D/b/RMSProp_1*
T0*!
_class
loc:@actor/Conv1D/b
�
*actor/Conv1D_1/W/RMSProp/Initializer/ConstConst*&
valueB�*  �?*#
_class
loc:@actor/Conv1D_1/W*
dtype0
�
actor/Conv1D_1/W/RMSProp
VariableV2*
shared_name *#
_class
loc:@actor/Conv1D_1/W*
dtype0*
	container *
shape:�
�
actor/Conv1D_1/W/RMSProp/AssignIdentity*actor/Conv1D_1/W/RMSProp/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_1/W
q
actor/Conv1D_1/W/RMSProp/readIdentityactor/Conv1D_1/W/RMSProp*
T0*#
_class
loc:@actor/Conv1D_1/W
�
,actor/Conv1D_1/W/RMSProp_1/Initializer/ConstConst*&
valueB�*    *#
_class
loc:@actor/Conv1D_1/W*
dtype0
�
actor/Conv1D_1/W/RMSProp_1
VariableV2*
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_1/W*
dtype0*
	container 
�
!actor/Conv1D_1/W/RMSProp_1/AssignIdentity,actor/Conv1D_1/W/RMSProp_1/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_1/W
u
actor/Conv1D_1/W/RMSProp_1/readIdentityactor/Conv1D_1/W/RMSProp_1*
T0*#
_class
loc:@actor/Conv1D_1/W
�
*actor/Conv1D_1/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*#
_class
loc:@actor/Conv1D_1/b*
dtype0
�
actor/Conv1D_1/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_1/b
�
actor/Conv1D_1/b/RMSProp/AssignIdentity*actor/Conv1D_1/b/RMSProp/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_1/b
q
actor/Conv1D_1/b/RMSProp/readIdentityactor/Conv1D_1/b/RMSProp*
T0*#
_class
loc:@actor/Conv1D_1/b
�
,actor/Conv1D_1/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *#
_class
loc:@actor/Conv1D_1/b*
dtype0
�
actor/Conv1D_1/b/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_1/b
�
!actor/Conv1D_1/b/RMSProp_1/AssignIdentity,actor/Conv1D_1/b/RMSProp_1/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_1/b
u
actor/Conv1D_1/b/RMSProp_1/readIdentityactor/Conv1D_1/b/RMSProp_1*
T0*#
_class
loc:@actor/Conv1D_1/b
�
*actor/Conv1D_2/W/RMSProp/Initializer/ConstConst*&
valueB�*  �?*#
_class
loc:@actor/Conv1D_2/W*
dtype0
�
actor/Conv1D_2/W/RMSProp
VariableV2*#
_class
loc:@actor/Conv1D_2/W*
dtype0*
	container *
shape:�*
shared_name 
�
actor/Conv1D_2/W/RMSProp/AssignIdentity*actor/Conv1D_2/W/RMSProp/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_2/W
q
actor/Conv1D_2/W/RMSProp/readIdentityactor/Conv1D_2/W/RMSProp*
T0*#
_class
loc:@actor/Conv1D_2/W
�
,actor/Conv1D_2/W/RMSProp_1/Initializer/ConstConst*
dtype0*&
valueB�*    *#
_class
loc:@actor/Conv1D_2/W
�
actor/Conv1D_2/W/RMSProp_1
VariableV2*#
_class
loc:@actor/Conv1D_2/W*
dtype0*
	container *
shape:�*
shared_name 
�
!actor/Conv1D_2/W/RMSProp_1/AssignIdentity,actor/Conv1D_2/W/RMSProp_1/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_2/W
u
actor/Conv1D_2/W/RMSProp_1/readIdentityactor/Conv1D_2/W/RMSProp_1*
T0*#
_class
loc:@actor/Conv1D_2/W
�
*actor/Conv1D_2/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*#
_class
loc:@actor/Conv1D_2/b
�
actor/Conv1D_2/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_2/b
�
actor/Conv1D_2/b/RMSProp/AssignIdentity*actor/Conv1D_2/b/RMSProp/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_2/b
q
actor/Conv1D_2/b/RMSProp/readIdentityactor/Conv1D_2/b/RMSProp*
T0*#
_class
loc:@actor/Conv1D_2/b
�
,actor/Conv1D_2/b/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB�*    *#
_class
loc:@actor/Conv1D_2/b
�
actor/Conv1D_2/b/RMSProp_1
VariableV2*
shape:�*
shared_name *#
_class
loc:@actor/Conv1D_2/b*
dtype0*
	container 
�
!actor/Conv1D_2/b/RMSProp_1/AssignIdentity,actor/Conv1D_2/b/RMSProp_1/Initializer/Const*
T0*#
_class
loc:@actor/Conv1D_2/b
u
actor/Conv1D_2/b/RMSProp_1/readIdentityactor/Conv1D_2/b/RMSProp_1*
T0*#
_class
loc:@actor/Conv1D_2/b
�
2actor/FullyConnected_2/W/RMSProp/Initializer/ConstConst*
valueB	�*  �?*+
_class!
loc:@actor/FullyConnected_2/W*
dtype0
�
 actor/FullyConnected_2/W/RMSProp
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *+
_class!
loc:@actor/FullyConnected_2/W
�
'actor/FullyConnected_2/W/RMSProp/AssignIdentity2actor/FullyConnected_2/W/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
%actor/FullyConnected_2/W/RMSProp/readIdentity actor/FullyConnected_2/W/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
4actor/FullyConnected_2/W/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB	�*    *+
_class!
loc:@actor/FullyConnected_2/W
�
"actor/FullyConnected_2/W/RMSProp_1
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_2/W*
dtype0*
	container *
shape:	�
�
)actor/FullyConnected_2/W/RMSProp_1/AssignIdentity4actor/FullyConnected_2/W/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
'actor/FullyConnected_2/W/RMSProp_1/readIdentity"actor/FullyConnected_2/W/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_2/W
�
2actor/FullyConnected_2/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*+
_class!
loc:@actor/FullyConnected_2/b
�
 actor/FullyConnected_2/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *+
_class!
loc:@actor/FullyConnected_2/b
�
'actor/FullyConnected_2/b/RMSProp/AssignIdentity2actor/FullyConnected_2/b/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
%actor/FullyConnected_2/b/RMSProp/readIdentity actor/FullyConnected_2/b/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
4actor/FullyConnected_2/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *+
_class!
loc:@actor/FullyConnected_2/b*
dtype0
�
"actor/FullyConnected_2/b/RMSProp_1
VariableV2*+
_class!
loc:@actor/FullyConnected_2/b*
dtype0*
	container *
shape:�*
shared_name 
�
)actor/FullyConnected_2/b/RMSProp_1/AssignIdentity4actor/FullyConnected_2/b/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
'actor/FullyConnected_2/b/RMSProp_1/readIdentity"actor/FullyConnected_2/b/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
2actor/FullyConnected_3/W/RMSProp/Initializer/ConstConst*
dtype0*
valueB
��*  �?*+
_class!
loc:@actor/FullyConnected_3/W
�
 actor/FullyConnected_3/W/RMSProp
VariableV2*
shape:
��*
shared_name *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0*
	container 
�
'actor/FullyConnected_3/W/RMSProp/AssignIdentity2actor/FullyConnected_3/W/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
%actor/FullyConnected_3/W/RMSProp/readIdentity actor/FullyConnected_3/W/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
4actor/FullyConnected_3/W/RMSProp_1/Initializer/ConstConst*
valueB
��*    *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0
�
"actor/FullyConnected_3/W/RMSProp_1
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_3/W*
dtype0*
	container *
shape:
��
�
)actor/FullyConnected_3/W/RMSProp_1/AssignIdentity4actor/FullyConnected_3/W/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
'actor/FullyConnected_3/W/RMSProp_1/readIdentity"actor/FullyConnected_3/W/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
2actor/FullyConnected_3/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*+
_class!
loc:@actor/FullyConnected_3/b*
dtype0
�
 actor/FullyConnected_3/b/RMSProp
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_3/b*
dtype0*
	container *
shape:�
�
'actor/FullyConnected_3/b/RMSProp/AssignIdentity2actor/FullyConnected_3/b/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
%actor/FullyConnected_3/b/RMSProp/readIdentity actor/FullyConnected_3/b/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
4actor/FullyConnected_3/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *+
_class!
loc:@actor/FullyConnected_3/b*
dtype0
�
"actor/FullyConnected_3/b/RMSProp_1
VariableV2*
shape:�*
shared_name *+
_class!
loc:@actor/FullyConnected_3/b*
dtype0*
	container 
�
)actor/FullyConnected_3/b/RMSProp_1/AssignIdentity4actor/FullyConnected_3/b/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
'actor/FullyConnected_3/b/RMSProp_1/readIdentity"actor/FullyConnected_3/b/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
2actor/FullyConnected_4/W/RMSProp/Initializer/ConstConst*
valueB	�*  �?*+
_class!
loc:@actor/FullyConnected_4/W*
dtype0
�
 actor/FullyConnected_4/W/RMSProp
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_4/W*
dtype0*
	container *
shape:	�
�
'actor/FullyConnected_4/W/RMSProp/AssignIdentity2actor/FullyConnected_4/W/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
%actor/FullyConnected_4/W/RMSProp/readIdentity actor/FullyConnected_4/W/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
4actor/FullyConnected_4/W/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB	�*    *+
_class!
loc:@actor/FullyConnected_4/W
�
"actor/FullyConnected_4/W/RMSProp_1
VariableV2*
shared_name *+
_class!
loc:@actor/FullyConnected_4/W*
dtype0*
	container *
shape:	�
�
)actor/FullyConnected_4/W/RMSProp_1/AssignIdentity4actor/FullyConnected_4/W/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
'actor/FullyConnected_4/W/RMSProp_1/readIdentity"actor/FullyConnected_4/W/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
2actor/FullyConnected_4/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB*  �?*+
_class!
loc:@actor/FullyConnected_4/b
�
 actor/FullyConnected_4/b/RMSProp
VariableV2*+
_class!
loc:@actor/FullyConnected_4/b*
dtype0*
	container *
shape:*
shared_name 
�
'actor/FullyConnected_4/b/RMSProp/AssignIdentity2actor/FullyConnected_4/b/RMSProp/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_4/b
�
%actor/FullyConnected_4/b/RMSProp/readIdentity actor/FullyConnected_4/b/RMSProp*
T0*+
_class!
loc:@actor/FullyConnected_4/b
�
4actor/FullyConnected_4/b/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB*    *+
_class!
loc:@actor/FullyConnected_4/b
�
"actor/FullyConnected_4/b/RMSProp_1
VariableV2*
shape:*
shared_name *+
_class!
loc:@actor/FullyConnected_4/b*
dtype0*
	container 
�
)actor/FullyConnected_4/b/RMSProp_1/AssignIdentity4actor/FullyConnected_4/b/RMSProp_1/Initializer/Const*
T0*+
_class!
loc:@actor/FullyConnected_4/b
�
'actor/FullyConnected_4/b/RMSProp_1/readIdentity"actor/FullyConnected_4/b/RMSProp_1*
T0*+
_class!
loc:@actor/FullyConnected_4/b
B
RMSProp/learning_rateConst*
dtype0*
valueB
 *��8
:
RMSProp/decayConst*
valueB
 *fff?*
dtype0
=
RMSProp/momentumConst*
valueB
 *    *
dtype0
<
RMSProp/epsilonConst*
valueB
 *���.*
dtype0
�
2RMSProp/update_actor/FullyConnected/W/ApplyRMSPropApplyRMSPropactor/FullyConnected/Wactor/FullyConnected/W/RMSProp actor/FullyConnected/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon3gradients/actor/FullyConnected/MatMul_grad/MatMul_1*
T0*)
_class
loc:@actor/FullyConnected/W*
use_locking( 
�
2RMSProp/update_actor/FullyConnected/b/ApplyRMSPropApplyRMSPropactor/FullyConnected/bactor/FullyConnected/b/RMSProp actor/FullyConnected/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon7gradients/actor/FullyConnected/BiasAdd_grad/BiasAddGrad*
T0*)
_class
loc:@actor/FullyConnected/b*
use_locking( 
�
4RMSProp/update_actor/FullyConnected_1/W/ApplyRMSPropApplyRMSPropactor/FullyConnected_1/W actor/FullyConnected_1/W/RMSProp"actor/FullyConnected_1/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon5gradients/actor/FullyConnected_1/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_1/W
�
4RMSProp/update_actor/FullyConnected_1/b/ApplyRMSPropApplyRMSPropactor/FullyConnected_1/b actor/FullyConnected_1/b/RMSProp"actor/FullyConnected_1/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_1/b
�
*RMSProp/update_actor/Conv1D/W/ApplyRMSPropApplyRMSPropactor/Conv1D/Wactor/Conv1D/W/RMSPropactor/Conv1D/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon7gradients/actor/Conv1D/Conv2D_grad/Conv2DBackpropFilter*
use_locking( *
T0*!
_class
loc:@actor/Conv1D/W
�
*RMSProp/update_actor/Conv1D/b/ApplyRMSPropApplyRMSPropactor/Conv1D/bactor/Conv1D/b/RMSPropactor/Conv1D/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon/gradients/actor/Conv1D/BiasAdd_grad/BiasAddGrad*
T0*!
_class
loc:@actor/Conv1D/b*
use_locking( 
�
,RMSProp/update_actor/Conv1D_1/W/ApplyRMSPropApplyRMSPropactor/Conv1D_1/Wactor/Conv1D_1/W/RMSPropactor/Conv1D_1/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/Conv1D_1/Conv2D_grad/Conv2DBackpropFilter*
use_locking( *
T0*#
_class
loc:@actor/Conv1D_1/W
�
,RMSProp/update_actor/Conv1D_1/b/ApplyRMSPropApplyRMSPropactor/Conv1D_1/bactor/Conv1D_1/b/RMSPropactor/Conv1D_1/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon1gradients/actor/Conv1D_1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*#
_class
loc:@actor/Conv1D_1/b
�
,RMSProp/update_actor/Conv1D_2/W/ApplyRMSPropApplyRMSPropactor/Conv1D_2/Wactor/Conv1D_2/W/RMSPropactor/Conv1D_2/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/Conv1D_2/Conv2D_grad/Conv2DBackpropFilter*
use_locking( *
T0*#
_class
loc:@actor/Conv1D_2/W
�
,RMSProp/update_actor/Conv1D_2/b/ApplyRMSPropApplyRMSPropactor/Conv1D_2/bactor/Conv1D_2/b/RMSPropactor/Conv1D_2/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon1gradients/actor/Conv1D_2/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*#
_class
loc:@actor/Conv1D_2/b
�
4RMSProp/update_actor/FullyConnected_2/W/ApplyRMSPropApplyRMSPropactor/FullyConnected_2/W actor/FullyConnected_2/W/RMSProp"actor/FullyConnected_2/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon5gradients/actor/FullyConnected_2/MatMul_grad/MatMul_1*
T0*+
_class!
loc:@actor/FullyConnected_2/W*
use_locking( 
�
4RMSProp/update_actor/FullyConnected_2/b/ApplyRMSPropApplyRMSPropactor/FullyConnected_2/b actor/FullyConnected_2/b/RMSProp"actor/FullyConnected_2/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/FullyConnected_2/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_2/b
�
4RMSProp/update_actor/FullyConnected_3/W/ApplyRMSPropApplyRMSPropactor/FullyConnected_3/W actor/FullyConnected_3/W/RMSProp"actor/FullyConnected_3/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon5gradients/actor/FullyConnected_3/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_3/W
�
4RMSProp/update_actor/FullyConnected_3/b/ApplyRMSPropApplyRMSPropactor/FullyConnected_3/b actor/FullyConnected_3/b/RMSProp"actor/FullyConnected_3/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/FullyConnected_3/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_3/b
�
4RMSProp/update_actor/FullyConnected_4/W/ApplyRMSPropApplyRMSPropactor/FullyConnected_4/W actor/FullyConnected_4/W/RMSProp"actor/FullyConnected_4/W/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon5gradients/actor/FullyConnected_4/MatMul_grad/MatMul_1*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_4/W
�
4RMSProp/update_actor/FullyConnected_4/b/ApplyRMSPropApplyRMSPropactor/FullyConnected_4/b actor/FullyConnected_4/b/RMSProp"actor/FullyConnected_4/b/RMSProp_1RMSProp/learning_rateRMSProp/decayRMSProp/momentumRMSProp/epsilon9gradients/actor/FullyConnected_4/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*+
_class!
loc:@actor/FullyConnected_4/b
�
RMSPropNoOp+^RMSProp/update_actor/Conv1D/W/ApplyRMSProp+^RMSProp/update_actor/Conv1D/b/ApplyRMSProp-^RMSProp/update_actor/Conv1D_1/W/ApplyRMSProp-^RMSProp/update_actor/Conv1D_1/b/ApplyRMSProp-^RMSProp/update_actor/Conv1D_2/W/ApplyRMSProp-^RMSProp/update_actor/Conv1D_2/b/ApplyRMSProp3^RMSProp/update_actor/FullyConnected/W/ApplyRMSProp3^RMSProp/update_actor/FullyConnected/b/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_1/W/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_1/b/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_2/W/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_2/b/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_3/W/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_3/b/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_4/W/ApplyRMSProp5^RMSProp/update_actor/FullyConnected_4/b/ApplyRMSProp
;
critic/InputData/XPlaceholder*
dtype0*
shape: 
S
critic/strided_slice/stackConst*!
valueB"        ����*
dtype0
U
critic/strided_slice/stack_1Const*!
valueB"           *
dtype0
U
critic/strided_slice/stack_2Const*!
valueB"         *
dtype0
�
critic/strided_sliceStridedSlicecritic/InputData/Xcritic/strided_slice/stackcritic/strided_slice/stack_1critic/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
�
:critic/FullyConnected/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"   �   **
_class 
loc:@critic/FullyConnected/W
�
9critic/FullyConnected/W/Initializer/truncated_normal/meanConst*
valueB
 *    **
_class 
loc:@critic/FullyConnected/W*
dtype0
�
;critic/FullyConnected/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<**
_class 
loc:@critic/FullyConnected/W*
dtype0
�
Dcritic/FullyConnected/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal:critic/FullyConnected/W/Initializer/truncated_normal/shape*
T0**
_class 
loc:@critic/FullyConnected/W*
dtype0*
seed2 *

seed 
�
8critic/FullyConnected/W/Initializer/truncated_normal/mulMulDcritic/FullyConnected/W/Initializer/truncated_normal/TruncatedNormal;critic/FullyConnected/W/Initializer/truncated_normal/stddev*
T0**
_class 
loc:@critic/FullyConnected/W
�
4critic/FullyConnected/W/Initializer/truncated_normalAdd8critic/FullyConnected/W/Initializer/truncated_normal/mul9critic/FullyConnected/W/Initializer/truncated_normal/mean*
T0**
_class 
loc:@critic/FullyConnected/W
�
critic/FullyConnected/W
VariableV2*
dtype0*
	container *
shape:	�*
shared_name **
_class 
loc:@critic/FullyConnected/W
�
critic/FullyConnected/W/AssignIdentity4critic/FullyConnected/W/Initializer/truncated_normal*
T0**
_class 
loc:@critic/FullyConnected/W
v
critic/FullyConnected/W/readIdentitycritic/FullyConnected/W*
T0**
_class 
loc:@critic/FullyConnected/W
�
)critic/FullyConnected/b/Initializer/ConstConst*
dtype0*
valueB�*    **
_class 
loc:@critic/FullyConnected/b
�
critic/FullyConnected/b
VariableV2*
shape:�*
shared_name **
_class 
loc:@critic/FullyConnected/b*
dtype0*
	container 
�
critic/FullyConnected/b/AssignIdentity)critic/FullyConnected/b/Initializer/Const*
T0**
_class 
loc:@critic/FullyConnected/b
v
critic/FullyConnected/b/readIdentitycritic/FullyConnected/b*
T0**
_class 
loc:@critic/FullyConnected/b
�
critic/FullyConnected/MatMulMatMulcritic/strided_slicecritic/FullyConnected/W/read*
T0*
transpose_a( *
transpose_b( 
�
critic/FullyConnected/BiasAddBiasAddcritic/FullyConnected/MatMulcritic/FullyConnected/b/read*
T0*
data_formatNHWC
J
critic/FullyConnected/ReluRelucritic/FullyConnected/BiasAdd*
T0
U
critic/strided_slice_1/stackConst*!
valueB"       ����*
dtype0
W
critic/strided_slice_1/stack_1Const*!
valueB"           *
dtype0
W
critic/strided_slice_1/stack_2Const*!
valueB"         *
dtype0
�
critic/strided_slice_1StridedSlicecritic/InputData/Xcritic/strided_slice_1/stackcritic/strided_slice_1/stack_1critic/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
T0*
Index0
�
<critic/FullyConnected_1/W/Initializer/truncated_normal/shapeConst*
valueB"   �   *,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0
�
;critic/FullyConnected_1/W/Initializer/truncated_normal/meanConst*
valueB
 *    *,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0
�
=critic/FullyConnected_1/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0
�
Fcritic/FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal<critic/FullyConnected_1/W/Initializer/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
:critic/FullyConnected_1/W/Initializer/truncated_normal/mulMulFcritic/FullyConnected_1/W/Initializer/truncated_normal/TruncatedNormal=critic/FullyConnected_1/W/Initializer/truncated_normal/stddev*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
6critic/FullyConnected_1/W/Initializer/truncated_normalAdd:critic/FullyConnected_1/W/Initializer/truncated_normal/mul;critic/FullyConnected_1/W/Initializer/truncated_normal/mean*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
critic/FullyConnected_1/W
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_1/W
�
 critic/FullyConnected_1/W/AssignIdentity6critic/FullyConnected_1/W/Initializer/truncated_normal*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
|
critic/FullyConnected_1/W/readIdentitycritic/FullyConnected_1/W*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
+critic/FullyConnected_1/b/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_1/b*
dtype0
�
critic/FullyConnected_1/b
VariableV2*
shape:�*
shared_name *,
_class"
 loc:@critic/FullyConnected_1/b*
dtype0*
	container 
�
 critic/FullyConnected_1/b/AssignIdentity+critic/FullyConnected_1/b/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
|
critic/FullyConnected_1/b/readIdentitycritic/FullyConnected_1/b*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
critic/FullyConnected_1/MatMulMatMulcritic/strided_slice_1critic/FullyConnected_1/W/read*
T0*
transpose_a( *
transpose_b( 
�
critic/FullyConnected_1/BiasAddBiasAddcritic/FullyConnected_1/MatMulcritic/FullyConnected_1/b/read*
T0*
data_formatNHWC
N
critic/FullyConnected_1/ReluRelucritic/FullyConnected_1/BiasAdd*
T0
U
critic/strided_slice_2/stackConst*!
valueB"           *
dtype0
W
critic/strided_slice_2/stack_1Const*
dtype0*!
valueB"           
W
critic/strided_slice_2/stack_2Const*!
valueB"         *
dtype0
�
critic/strided_slice_2StridedSlicecritic/InputData/Xcritic/strided_slice_2/stackcritic/strided_slice_2/stack_1critic/strided_slice_2/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
T0*
Index0
�
0critic/Conv1D/W/Initializer/random_uniform/shapeConst*%
valueB"         �   *"
_class
loc:@critic/Conv1D/W*
dtype0

.critic/Conv1D/W/Initializer/random_uniform/minConst*
dtype0*
valueB
 *qĜ�*"
_class
loc:@critic/Conv1D/W

.critic/Conv1D/W/Initializer/random_uniform/maxConst*
valueB
 *qĜ>*"
_class
loc:@critic/Conv1D/W*
dtype0
�
8critic/Conv1D/W/Initializer/random_uniform/RandomUniformRandomUniform0critic/Conv1D/W/Initializer/random_uniform/shape*
T0*"
_class
loc:@critic/Conv1D/W*
dtype0*
seed2 *

seed 
�
.critic/Conv1D/W/Initializer/random_uniform/subSub.critic/Conv1D/W/Initializer/random_uniform/max.critic/Conv1D/W/Initializer/random_uniform/min*
T0*"
_class
loc:@critic/Conv1D/W
�
.critic/Conv1D/W/Initializer/random_uniform/mulMul8critic/Conv1D/W/Initializer/random_uniform/RandomUniform.critic/Conv1D/W/Initializer/random_uniform/sub*
T0*"
_class
loc:@critic/Conv1D/W
�
*critic/Conv1D/W/Initializer/random_uniformAdd.critic/Conv1D/W/Initializer/random_uniform/mul.critic/Conv1D/W/Initializer/random_uniform/min*
T0*"
_class
loc:@critic/Conv1D/W
�
critic/Conv1D/W
VariableV2*"
_class
loc:@critic/Conv1D/W*
dtype0*
	container *
shape:�*
shared_name 
{
critic/Conv1D/W/AssignIdentity*critic/Conv1D/W/Initializer/random_uniform*
T0*"
_class
loc:@critic/Conv1D/W
^
critic/Conv1D/W/readIdentitycritic/Conv1D/W*
T0*"
_class
loc:@critic/Conv1D/W
w
!critic/Conv1D/b/Initializer/ConstConst*
dtype0*
valueB�*    *"
_class
loc:@critic/Conv1D/b
�
critic/Conv1D/b
VariableV2*
shared_name *"
_class
loc:@critic/Conv1D/b*
dtype0*
	container *
shape:�
r
critic/Conv1D/b/AssignIdentity!critic/Conv1D/b/Initializer/Const*
T0*"
_class
loc:@critic/Conv1D/b
^
critic/Conv1D/b/readIdentitycritic/Conv1D/b*
T0*"
_class
loc:@critic/Conv1D/b
F
critic/Conv1D/ExpandDims/dimConst*
value	B :*
dtype0
q
critic/Conv1D/ExpandDims
ExpandDimscritic/strided_slice_2critic/Conv1D/ExpandDims/dim*

Tdim0*
T0
�
critic/Conv1D/Conv2DConv2Dcritic/Conv1D/ExpandDimscritic/Conv1D/W/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

l
critic/Conv1D/BiasAddBiasAddcritic/Conv1D/Conv2Dcritic/Conv1D/b/read*
data_formatNHWC*
T0
W
critic/Conv1D/SqueezeSqueezecritic/Conv1D/BiasAdd*
squeeze_dims
*
T0
:
critic/Conv1D/ReluRelucritic/Conv1D/Squeeze*
T0
U
critic/strided_slice_3/stackConst*!
valueB"           *
dtype0
W
critic/strided_slice_3/stack_1Const*!
valueB"           *
dtype0
W
critic/strided_slice_3/stack_2Const*!
valueB"         *
dtype0
�
critic/strided_slice_3StridedSlicecritic/InputData/Xcritic/strided_slice_3/stackcritic/strided_slice_3/stack_1critic/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0
�
2critic/Conv1D_1/W/Initializer/random_uniform/shapeConst*%
valueB"         �   *$
_class
loc:@critic/Conv1D_1/W*
dtype0
�
0critic/Conv1D_1/W/Initializer/random_uniform/minConst*
valueB
 *qĜ�*$
_class
loc:@critic/Conv1D_1/W*
dtype0
�
0critic/Conv1D_1/W/Initializer/random_uniform/maxConst*
valueB
 *qĜ>*$
_class
loc:@critic/Conv1D_1/W*
dtype0
�
:critic/Conv1D_1/W/Initializer/random_uniform/RandomUniformRandomUniform2critic/Conv1D_1/W/Initializer/random_uniform/shape*

seed *
T0*$
_class
loc:@critic/Conv1D_1/W*
dtype0*
seed2 
�
0critic/Conv1D_1/W/Initializer/random_uniform/subSub0critic/Conv1D_1/W/Initializer/random_uniform/max0critic/Conv1D_1/W/Initializer/random_uniform/min*
T0*$
_class
loc:@critic/Conv1D_1/W
�
0critic/Conv1D_1/W/Initializer/random_uniform/mulMul:critic/Conv1D_1/W/Initializer/random_uniform/RandomUniform0critic/Conv1D_1/W/Initializer/random_uniform/sub*
T0*$
_class
loc:@critic/Conv1D_1/W
�
,critic/Conv1D_1/W/Initializer/random_uniformAdd0critic/Conv1D_1/W/Initializer/random_uniform/mul0critic/Conv1D_1/W/Initializer/random_uniform/min*
T0*$
_class
loc:@critic/Conv1D_1/W
�
critic/Conv1D_1/W
VariableV2*
shared_name *$
_class
loc:@critic/Conv1D_1/W*
dtype0*
	container *
shape:�
�
critic/Conv1D_1/W/AssignIdentity,critic/Conv1D_1/W/Initializer/random_uniform*
T0*$
_class
loc:@critic/Conv1D_1/W
d
critic/Conv1D_1/W/readIdentitycritic/Conv1D_1/W*
T0*$
_class
loc:@critic/Conv1D_1/W
{
#critic/Conv1D_1/b/Initializer/ConstConst*
valueB�*    *$
_class
loc:@critic/Conv1D_1/b*
dtype0
�
critic/Conv1D_1/b
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_1/b
x
critic/Conv1D_1/b/AssignIdentity#critic/Conv1D_1/b/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_1/b
d
critic/Conv1D_1/b/readIdentitycritic/Conv1D_1/b*
T0*$
_class
loc:@critic/Conv1D_1/b
H
critic/Conv1D_1/ExpandDims/dimConst*
value	B :*
dtype0
u
critic/Conv1D_1/ExpandDims
ExpandDimscritic/strided_slice_3critic/Conv1D_1/ExpandDims/dim*

Tdim0*
T0
�
critic/Conv1D_1/Conv2DConv2Dcritic/Conv1D_1/ExpandDimscritic/Conv1D_1/W/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
r
critic/Conv1D_1/BiasAddBiasAddcritic/Conv1D_1/Conv2Dcritic/Conv1D_1/b/read*
T0*
data_formatNHWC
[
critic/Conv1D_1/SqueezeSqueezecritic/Conv1D_1/BiasAdd*
squeeze_dims
*
T0
>
critic/Conv1D_1/ReluRelucritic/Conv1D_1/Squeeze*
T0
U
critic/strided_slice_4/stackConst*!
valueB"           *
dtype0
W
critic/strided_slice_4/stack_1Const*!
valueB"          *
dtype0
W
critic/strided_slice_4/stack_2Const*!
valueB"         *
dtype0
�
critic/strided_slice_4StridedSlicecritic/InputData/Xcritic/strided_slice_4/stackcritic/strided_slice_4/stack_1critic/strided_slice_4/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
Index0*
T0*
shrink_axis_mask 
�
2critic/Conv1D_2/W/Initializer/random_uniform/shapeConst*%
valueB"         �   *$
_class
loc:@critic/Conv1D_2/W*
dtype0
�
0critic/Conv1D_2/W/Initializer/random_uniform/minConst*
valueB
 *���*$
_class
loc:@critic/Conv1D_2/W*
dtype0
�
0critic/Conv1D_2/W/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *��>*$
_class
loc:@critic/Conv1D_2/W
�
:critic/Conv1D_2/W/Initializer/random_uniform/RandomUniformRandomUniform2critic/Conv1D_2/W/Initializer/random_uniform/shape*
T0*$
_class
loc:@critic/Conv1D_2/W*
dtype0*
seed2 *

seed 
�
0critic/Conv1D_2/W/Initializer/random_uniform/subSub0critic/Conv1D_2/W/Initializer/random_uniform/max0critic/Conv1D_2/W/Initializer/random_uniform/min*
T0*$
_class
loc:@critic/Conv1D_2/W
�
0critic/Conv1D_2/W/Initializer/random_uniform/mulMul:critic/Conv1D_2/W/Initializer/random_uniform/RandomUniform0critic/Conv1D_2/W/Initializer/random_uniform/sub*
T0*$
_class
loc:@critic/Conv1D_2/W
�
,critic/Conv1D_2/W/Initializer/random_uniformAdd0critic/Conv1D_2/W/Initializer/random_uniform/mul0critic/Conv1D_2/W/Initializer/random_uniform/min*
T0*$
_class
loc:@critic/Conv1D_2/W
�
critic/Conv1D_2/W
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_2/W
�
critic/Conv1D_2/W/AssignIdentity,critic/Conv1D_2/W/Initializer/random_uniform*
T0*$
_class
loc:@critic/Conv1D_2/W
d
critic/Conv1D_2/W/readIdentitycritic/Conv1D_2/W*
T0*$
_class
loc:@critic/Conv1D_2/W
{
#critic/Conv1D_2/b/Initializer/ConstConst*
valueB�*    *$
_class
loc:@critic/Conv1D_2/b*
dtype0
�
critic/Conv1D_2/b
VariableV2*
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_2/b*
dtype0*
	container 
x
critic/Conv1D_2/b/AssignIdentity#critic/Conv1D_2/b/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_2/b
d
critic/Conv1D_2/b/readIdentitycritic/Conv1D_2/b*
T0*$
_class
loc:@critic/Conv1D_2/b
H
critic/Conv1D_2/ExpandDims/dimConst*
value	B :*
dtype0
u
critic/Conv1D_2/ExpandDims
ExpandDimscritic/strided_slice_4critic/Conv1D_2/ExpandDims/dim*
T0*

Tdim0
�
critic/Conv1D_2/Conv2DConv2Dcritic/Conv1D_2/ExpandDimscritic/Conv1D_2/W/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
r
critic/Conv1D_2/BiasAddBiasAddcritic/Conv1D_2/Conv2Dcritic/Conv1D_2/b/read*
data_formatNHWC*
T0
[
critic/Conv1D_2/SqueezeSqueezecritic/Conv1D_2/BiasAdd*
squeeze_dims
*
T0
>
critic/Conv1D_2/ReluRelucritic/Conv1D_2/Squeeze*
T0
U
critic/strided_slice_5/stackConst*
dtype0*!
valueB"       ����
W
critic/strided_slice_5/stack_1Const*!
valueB"           *
dtype0
W
critic/strided_slice_5/stack_2Const*
dtype0*!
valueB"         
�
critic/strided_slice_5StridedSlicecritic/InputData/Xcritic/strided_slice_5/stackcritic/strided_slice_5/stack_1critic/strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
�
<critic/FullyConnected_2/W/Initializer/truncated_normal/shapeConst*
valueB"   �   *,
_class"
 loc:@critic/FullyConnected_2/W*
dtype0
�
;critic/FullyConnected_2/W/Initializer/truncated_normal/meanConst*
valueB
 *    *,
_class"
 loc:@critic/FullyConnected_2/W*
dtype0
�
=critic/FullyConnected_2/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*,
_class"
 loc:@critic/FullyConnected_2/W*
dtype0
�
Fcritic/FullyConnected_2/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal<critic/FullyConnected_2/W/Initializer/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
:critic/FullyConnected_2/W/Initializer/truncated_normal/mulMulFcritic/FullyConnected_2/W/Initializer/truncated_normal/TruncatedNormal=critic/FullyConnected_2/W/Initializer/truncated_normal/stddev*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
6critic/FullyConnected_2/W/Initializer/truncated_normalAdd:critic/FullyConnected_2/W/Initializer/truncated_normal/mul;critic/FullyConnected_2/W/Initializer/truncated_normal/mean*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
critic/FullyConnected_2/W
VariableV2*,
_class"
 loc:@critic/FullyConnected_2/W*
dtype0*
	container *
shape:	�*
shared_name 
�
 critic/FullyConnected_2/W/AssignIdentity6critic/FullyConnected_2/W/Initializer/truncated_normal*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
|
critic/FullyConnected_2/W/readIdentitycritic/FullyConnected_2/W*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
+critic/FullyConnected_2/b/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0
�
critic/FullyConnected_2/b
VariableV2*
shape:�*
shared_name *,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0*
	container 
�
 critic/FullyConnected_2/b/AssignIdentity+critic/FullyConnected_2/b/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
|
critic/FullyConnected_2/b/readIdentitycritic/FullyConnected_2/b*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
�
critic/FullyConnected_2/MatMulMatMulcritic/strided_slice_5critic/FullyConnected_2/W/read*
transpose_b( *
T0*
transpose_a( 
�
critic/FullyConnected_2/BiasAddBiasAddcritic/FullyConnected_2/MatMulcritic/FullyConnected_2/b/read*
T0*
data_formatNHWC
N
critic/FullyConnected_2/ReluRelucritic/FullyConnected_2/BiasAdd*
T0
Q
critic/Flatten/Reshape/shapeConst*
dtype0*
valueB"�����   
j
critic/Flatten/ReshapeReshapecritic/Conv1D/Relucritic/Flatten/Reshape/shape*
T0*
Tshape0
S
critic/Flatten_1/Reshape/shapeConst*
valueB"�����   *
dtype0
p
critic/Flatten_1/ReshapeReshapecritic/Conv1D_1/Relucritic/Flatten_1/Reshape/shape*
T0*
Tshape0
S
critic/Flatten_2/Reshape/shapeConst*
valueB"�����   *
dtype0
p
critic/Flatten_2/ReshapeReshapecritic/Conv1D_2/Relucritic/Flatten_2/Reshape/shape*
T0*
Tshape0
B
critic/Merge/concat/axisConst*
dtype0*
value	B :
�
critic/Merge/concatConcatV2critic/FullyConnected/Relucritic/FullyConnected_1/Relucritic/Flatten/Reshapecritic/Flatten_1/Reshapecritic/Flatten_2/Reshapecritic/FullyConnected_2/Relucritic/Merge/concat/axis*
T0*
N*

Tidx0
�
<critic/FullyConnected_3/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"   �   *,
_class"
 loc:@critic/FullyConnected_3/W
�
;critic/FullyConnected_3/W/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *,
_class"
 loc:@critic/FullyConnected_3/W
�
=critic/FullyConnected_3/W/Initializer/truncated_normal/stddevConst*
valueB
 *
ף<*,
_class"
 loc:@critic/FullyConnected_3/W*
dtype0
�
Fcritic/FullyConnected_3/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal<critic/FullyConnected_3/W/Initializer/truncated_normal/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
:critic/FullyConnected_3/W/Initializer/truncated_normal/mulMulFcritic/FullyConnected_3/W/Initializer/truncated_normal/TruncatedNormal=critic/FullyConnected_3/W/Initializer/truncated_normal/stddev*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
6critic/FullyConnected_3/W/Initializer/truncated_normalAdd:critic/FullyConnected_3/W/Initializer/truncated_normal/mul;critic/FullyConnected_3/W/Initializer/truncated_normal/mean*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
critic/FullyConnected_3/W
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *,
_class"
 loc:@critic/FullyConnected_3/W
�
 critic/FullyConnected_3/W/AssignIdentity6critic/FullyConnected_3/W/Initializer/truncated_normal*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
|
critic/FullyConnected_3/W/readIdentitycritic/FullyConnected_3/W*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
+critic/FullyConnected_3/b/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_3/b*
dtype0
�
critic/FullyConnected_3/b
VariableV2*,
_class"
 loc:@critic/FullyConnected_3/b*
dtype0*
	container *
shape:�*
shared_name 
�
 critic/FullyConnected_3/b/AssignIdentity+critic/FullyConnected_3/b/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
|
critic/FullyConnected_3/b/readIdentitycritic/FullyConnected_3/b*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
critic/FullyConnected_3/MatMulMatMulcritic/Merge/concatcritic/FullyConnected_3/W/read*
transpose_a( *
transpose_b( *
T0
�
critic/FullyConnected_3/BiasAddBiasAddcritic/FullyConnected_3/MatMulcritic/FullyConnected_3/b/read*
T0*
data_formatNHWC
N
critic/FullyConnected_3/ReluRelucritic/FullyConnected_3/BiasAdd*
T0
�
<critic/FullyConnected_4/W/Initializer/truncated_normal/shapeConst*
dtype0*
valueB"�      *,
_class"
 loc:@critic/FullyConnected_4/W
�
;critic/FullyConnected_4/W/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *,
_class"
 loc:@critic/FullyConnected_4/W
�
=critic/FullyConnected_4/W/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *
ף<*,
_class"
 loc:@critic/FullyConnected_4/W
�
Fcritic/FullyConnected_4/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal<critic/FullyConnected_4/W/Initializer/truncated_normal/shape*

seed *
T0*,
_class"
 loc:@critic/FullyConnected_4/W*
dtype0*
seed2 
�
:critic/FullyConnected_4/W/Initializer/truncated_normal/mulMulFcritic/FullyConnected_4/W/Initializer/truncated_normal/TruncatedNormal=critic/FullyConnected_4/W/Initializer/truncated_normal/stddev*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
6critic/FullyConnected_4/W/Initializer/truncated_normalAdd:critic/FullyConnected_4/W/Initializer/truncated_normal/mul;critic/FullyConnected_4/W/Initializer/truncated_normal/mean*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
critic/FullyConnected_4/W
VariableV2*
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/W*
dtype0*
	container 
�
 critic/FullyConnected_4/W/AssignIdentity6critic/FullyConnected_4/W/Initializer/truncated_normal*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
|
critic/FullyConnected_4/W/readIdentitycritic/FullyConnected_4/W*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
+critic/FullyConnected_4/b/Initializer/ConstConst*
valueB*    *,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0
�
critic/FullyConnected_4/b
VariableV2*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0*
	container *
shape:
�
 critic/FullyConnected_4/b/AssignIdentity+critic/FullyConnected_4/b/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
|
critic/FullyConnected_4/b/readIdentitycritic/FullyConnected_4/b*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
�
critic/FullyConnected_4/MatMulMatMulcritic/FullyConnected_3/Relucritic/FullyConnected_4/W/read*
T0*
transpose_a( *
transpose_b( 
�
critic/FullyConnected_4/BiasAddBiasAddcritic/FullyConnected_4/MatMulcritic/FullyConnected_4/b/read*
T0*
data_formatNHWC
@
Placeholder_18Placeholder*
dtype0*
shape:	�
<
Placeholder_19Placeholder*
dtype0*
shape:�
@
Placeholder_20Placeholder*
dtype0*
shape:	�
<
Placeholder_21Placeholder*
dtype0*
shape:�
H
Placeholder_22Placeholder*
dtype0*
shape:�
<
Placeholder_23Placeholder*
dtype0*
shape:�
H
Placeholder_24Placeholder*
dtype0*
shape:�
<
Placeholder_25Placeholder*
dtype0*
shape:�
H
Placeholder_26Placeholder*
dtype0*
shape:�
<
Placeholder_27Placeholder*
dtype0*
shape:�
@
Placeholder_28Placeholder*
shape:	�*
dtype0
<
Placeholder_29Placeholder*
shape:�*
dtype0
A
Placeholder_30Placeholder*
dtype0*
shape:
��
<
Placeholder_31Placeholder*
dtype0*
shape:�
@
Placeholder_32Placeholder*
dtype0*
shape:	�
;
Placeholder_33Placeholder*
dtype0*
shape:
Z
	Assign_18IdentityPlaceholder_18*
T0**
_class 
loc:@critic/FullyConnected/W
Z
	Assign_19IdentityPlaceholder_19*
T0**
_class 
loc:@critic/FullyConnected/b
\
	Assign_20IdentityPlaceholder_20*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
\
	Assign_21IdentityPlaceholder_21*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
R
	Assign_22IdentityPlaceholder_22*
T0*"
_class
loc:@critic/Conv1D/W
R
	Assign_23IdentityPlaceholder_23*
T0*"
_class
loc:@critic/Conv1D/b
T
	Assign_24IdentityPlaceholder_24*
T0*$
_class
loc:@critic/Conv1D_1/W
T
	Assign_25IdentityPlaceholder_25*
T0*$
_class
loc:@critic/Conv1D_1/b
T
	Assign_26IdentityPlaceholder_26*
T0*$
_class
loc:@critic/Conv1D_2/W
T
	Assign_27IdentityPlaceholder_27*
T0*$
_class
loc:@critic/Conv1D_2/b
\
	Assign_28IdentityPlaceholder_28*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
\
	Assign_29IdentityPlaceholder_29*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
\
	Assign_30IdentityPlaceholder_30*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
\
	Assign_31IdentityPlaceholder_31*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
\
	Assign_32IdentityPlaceholder_32*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
\
	Assign_33IdentityPlaceholder_33*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
7
Placeholder_34Placeholder*
shape: *
dtype0
D
SubSubPlaceholder_34critic/FullyConnected_4/BiasAdd*
T0
O
MeanSquare/subSubPlaceholder_34critic/FullyConnected_4/BiasAdd*
T0
4
MeanSquare/SquareSquareMeanSquare/sub*
T0
E
MeanSquare/ConstConst*
valueB"       *
dtype0
b
MeanSquare/MeanMeanMeanSquare/SquareMeanSquare/Const*
T0*

Tidx0*
	keep_dims( 
:
gradients_1/ShapeConst*
valueB *
dtype0
>
gradients_1/ConstConst*
valueB
 *  �?*
dtype0
Y
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*

index_type0
c
.gradients_1/MeanSquare/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
(gradients_1/MeanSquare/Mean_grad/ReshapeReshapegradients_1/Fill.gradients_1/MeanSquare/Mean_grad/Reshape/shape*
T0*
Tshape0
[
&gradients_1/MeanSquare/Mean_grad/ShapeShapeMeanSquare/Square*
T0*
out_type0
�
%gradients_1/MeanSquare/Mean_grad/TileTile(gradients_1/MeanSquare/Mean_grad/Reshape&gradients_1/MeanSquare/Mean_grad/Shape*
T0*

Tmultiples0
]
(gradients_1/MeanSquare/Mean_grad/Shape_1ShapeMeanSquare/Square*
T0*
out_type0
Q
(gradients_1/MeanSquare/Mean_grad/Shape_2Const*
valueB *
dtype0
T
&gradients_1/MeanSquare/Mean_grad/ConstConst*
valueB: *
dtype0
�
%gradients_1/MeanSquare/Mean_grad/ProdProd(gradients_1/MeanSquare/Mean_grad/Shape_1&gradients_1/MeanSquare/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( 
V
(gradients_1/MeanSquare/Mean_grad/Const_1Const*
valueB: *
dtype0
�
'gradients_1/MeanSquare/Mean_grad/Prod_1Prod(gradients_1/MeanSquare/Mean_grad/Shape_2(gradients_1/MeanSquare/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0
T
*gradients_1/MeanSquare/Mean_grad/Maximum/yConst*
value	B :*
dtype0
�
(gradients_1/MeanSquare/Mean_grad/MaximumMaximum'gradients_1/MeanSquare/Mean_grad/Prod_1*gradients_1/MeanSquare/Mean_grad/Maximum/y*
T0
�
)gradients_1/MeanSquare/Mean_grad/floordivFloorDiv%gradients_1/MeanSquare/Mean_grad/Prod(gradients_1/MeanSquare/Mean_grad/Maximum*
T0
�
%gradients_1/MeanSquare/Mean_grad/CastCast)gradients_1/MeanSquare/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0
�
(gradients_1/MeanSquare/Mean_grad/truedivRealDiv%gradients_1/MeanSquare/Mean_grad/Tile%gradients_1/MeanSquare/Mean_grad/Cast*
T0
�
(gradients_1/MeanSquare/Square_grad/mul/xConst)^gradients_1/MeanSquare/Mean_grad/truediv*
valueB
 *   @*
dtype0
p
&gradients_1/MeanSquare/Square_grad/mulMul(gradients_1/MeanSquare/Square_grad/mul/xMeanSquare/sub*
T0
�
(gradients_1/MeanSquare/Square_grad/mul_1Mul(gradients_1/MeanSquare/Mean_grad/truediv&gradients_1/MeanSquare/Square_grad/mul*
T0
W
%gradients_1/MeanSquare/sub_grad/ShapeShapePlaceholder_34*
T0*
out_type0
j
'gradients_1/MeanSquare/sub_grad/Shape_1Shapecritic/FullyConnected_4/BiasAdd*
T0*
out_type0
�
5gradients_1/MeanSquare/sub_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients_1/MeanSquare/sub_grad/Shape'gradients_1/MeanSquare/sub_grad/Shape_1*
T0
�
#gradients_1/MeanSquare/sub_grad/SumSum(gradients_1/MeanSquare/Square_grad/mul_15gradients_1/MeanSquare/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
'gradients_1/MeanSquare/sub_grad/ReshapeReshape#gradients_1/MeanSquare/sub_grad/Sum%gradients_1/MeanSquare/sub_grad/Shape*
T0*
Tshape0
�
%gradients_1/MeanSquare/sub_grad/Sum_1Sum(gradients_1/MeanSquare/Square_grad/mul_17gradients_1/MeanSquare/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Z
#gradients_1/MeanSquare/sub_grad/NegNeg%gradients_1/MeanSquare/sub_grad/Sum_1*
T0
�
)gradients_1/MeanSquare/sub_grad/Reshape_1Reshape#gradients_1/MeanSquare/sub_grad/Neg'gradients_1/MeanSquare/sub_grad/Shape_1*
T0*
Tshape0
�
<gradients_1/critic/FullyConnected_4/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/MeanSquare/sub_grad/Reshape_1*
T0*
data_formatNHWC
�
6gradients_1/critic/FullyConnected_4/MatMul_grad/MatMulMatMul)gradients_1/MeanSquare/sub_grad/Reshape_1critic/FullyConnected_4/W/read*
transpose_a( *
transpose_b(*
T0
�
8gradients_1/critic/FullyConnected_4/MatMul_grad/MatMul_1MatMulcritic/FullyConnected_3/Relu)gradients_1/MeanSquare/sub_grad/Reshape_1*
transpose_a(*
transpose_b( *
T0
�
6gradients_1/critic/FullyConnected_3/Relu_grad/ReluGradReluGrad6gradients_1/critic/FullyConnected_4/MatMul_grad/MatMulcritic/FullyConnected_3/Relu*
T0
�
<gradients_1/critic/FullyConnected_3/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/critic/FullyConnected_3/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMulMatMul6gradients_1/critic/FullyConnected_3/Relu_grad/ReluGradcritic/FullyConnected_3/W/read*
transpose_b(*
T0*
transpose_a( 
�
8gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul_1MatMulcritic/Merge/concat6gradients_1/critic/FullyConnected_3/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
S
)gradients_1/critic/Merge/concat_grad/RankConst*
dtype0*
value	B :
�
(gradients_1/critic/Merge/concat_grad/modFloorModcritic/Merge/concat/axis)gradients_1/critic/Merge/concat_grad/Rank*
T0
h
*gradients_1/critic/Merge/concat_grad/ShapeShapecritic/FullyConnected/Relu*
T0*
out_type0
�
+gradients_1/critic/Merge/concat_grad/ShapeNShapeNcritic/FullyConnected/Relucritic/FullyConnected_1/Relucritic/Flatten/Reshapecritic/Flatten_1/Reshapecritic/Flatten_2/Reshapecritic/FullyConnected_2/Relu*
T0*
out_type0*
N
�
1gradients_1/critic/Merge/concat_grad/ConcatOffsetConcatOffset(gradients_1/critic/Merge/concat_grad/mod+gradients_1/critic/Merge/concat_grad/ShapeN-gradients_1/critic/Merge/concat_grad/ShapeN:1-gradients_1/critic/Merge/concat_grad/ShapeN:2-gradients_1/critic/Merge/concat_grad/ShapeN:3-gradients_1/critic/Merge/concat_grad/ShapeN:4-gradients_1/critic/Merge/concat_grad/ShapeN:5*
N
�
*gradients_1/critic/Merge/concat_grad/SliceSlice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul1gradients_1/critic/Merge/concat_grad/ConcatOffset+gradients_1/critic/Merge/concat_grad/ShapeN*
T0*
Index0
�
,gradients_1/critic/Merge/concat_grad/Slice_1Slice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul3gradients_1/critic/Merge/concat_grad/ConcatOffset:1-gradients_1/critic/Merge/concat_grad/ShapeN:1*
T0*
Index0
�
,gradients_1/critic/Merge/concat_grad/Slice_2Slice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul3gradients_1/critic/Merge/concat_grad/ConcatOffset:2-gradients_1/critic/Merge/concat_grad/ShapeN:2*
T0*
Index0
�
,gradients_1/critic/Merge/concat_grad/Slice_3Slice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul3gradients_1/critic/Merge/concat_grad/ConcatOffset:3-gradients_1/critic/Merge/concat_grad/ShapeN:3*
T0*
Index0
�
,gradients_1/critic/Merge/concat_grad/Slice_4Slice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul3gradients_1/critic/Merge/concat_grad/ConcatOffset:4-gradients_1/critic/Merge/concat_grad/ShapeN:4*
T0*
Index0
�
,gradients_1/critic/Merge/concat_grad/Slice_5Slice6gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul3gradients_1/critic/Merge/concat_grad/ConcatOffset:5-gradients_1/critic/Merge/concat_grad/ShapeN:5*
T0*
Index0
�
4gradients_1/critic/FullyConnected/Relu_grad/ReluGradReluGrad*gradients_1/critic/Merge/concat_grad/Slicecritic/FullyConnected/Relu*
T0
�
6gradients_1/critic/FullyConnected_1/Relu_grad/ReluGradReluGrad,gradients_1/critic/Merge/concat_grad/Slice_1critic/FullyConnected_1/Relu*
T0
c
-gradients_1/critic/Flatten/Reshape_grad/ShapeShapecritic/Conv1D/Relu*
T0*
out_type0
�
/gradients_1/critic/Flatten/Reshape_grad/ReshapeReshape,gradients_1/critic/Merge/concat_grad/Slice_2-gradients_1/critic/Flatten/Reshape_grad/Shape*
T0*
Tshape0
g
/gradients_1/critic/Flatten_1/Reshape_grad/ShapeShapecritic/Conv1D_1/Relu*
T0*
out_type0
�
1gradients_1/critic/Flatten_1/Reshape_grad/ReshapeReshape,gradients_1/critic/Merge/concat_grad/Slice_3/gradients_1/critic/Flatten_1/Reshape_grad/Shape*
T0*
Tshape0
g
/gradients_1/critic/Flatten_2/Reshape_grad/ShapeShapecritic/Conv1D_2/Relu*
T0*
out_type0
�
1gradients_1/critic/Flatten_2/Reshape_grad/ReshapeReshape,gradients_1/critic/Merge/concat_grad/Slice_4/gradients_1/critic/Flatten_2/Reshape_grad/Shape*
T0*
Tshape0
�
6gradients_1/critic/FullyConnected_2/Relu_grad/ReluGradReluGrad,gradients_1/critic/Merge/concat_grad/Slice_5critic/FullyConnected_2/Relu*
T0
�
:gradients_1/critic/FullyConnected/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients_1/critic/FullyConnected/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
<gradients_1/critic/FullyConnected_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/critic/FullyConnected_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
,gradients_1/critic/Conv1D/Relu_grad/ReluGradReluGrad/gradients_1/critic/Flatten/Reshape_grad/Reshapecritic/Conv1D/Relu*
T0
�
.gradients_1/critic/Conv1D_1/Relu_grad/ReluGradReluGrad1gradients_1/critic/Flatten_1/Reshape_grad/Reshapecritic/Conv1D_1/Relu*
T0
�
.gradients_1/critic/Conv1D_2/Relu_grad/ReluGradReluGrad1gradients_1/critic/Flatten_2/Reshape_grad/Reshapecritic/Conv1D_2/Relu*
T0
�
<gradients_1/critic/FullyConnected_2/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients_1/critic/FullyConnected_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
4gradients_1/critic/FullyConnected/MatMul_grad/MatMulMatMul4gradients_1/critic/FullyConnected/Relu_grad/ReluGradcritic/FullyConnected/W/read*
transpose_b(*
T0*
transpose_a( 
�
6gradients_1/critic/FullyConnected/MatMul_grad/MatMul_1MatMulcritic/strided_slice4gradients_1/critic/FullyConnected/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
�
6gradients_1/critic/FullyConnected_1/MatMul_grad/MatMulMatMul6gradients_1/critic/FullyConnected_1/Relu_grad/ReluGradcritic/FullyConnected_1/W/read*
T0*
transpose_a( *
transpose_b(
�
8gradients_1/critic/FullyConnected_1/MatMul_grad/MatMul_1MatMulcritic/strided_slice_16gradients_1/critic/FullyConnected_1/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( 
e
,gradients_1/critic/Conv1D/Squeeze_grad/ShapeShapecritic/Conv1D/BiasAdd*
T0*
out_type0
�
.gradients_1/critic/Conv1D/Squeeze_grad/ReshapeReshape,gradients_1/critic/Conv1D/Relu_grad/ReluGrad,gradients_1/critic/Conv1D/Squeeze_grad/Shape*
T0*
Tshape0
i
.gradients_1/critic/Conv1D_1/Squeeze_grad/ShapeShapecritic/Conv1D_1/BiasAdd*
T0*
out_type0
�
0gradients_1/critic/Conv1D_1/Squeeze_grad/ReshapeReshape.gradients_1/critic/Conv1D_1/Relu_grad/ReluGrad.gradients_1/critic/Conv1D_1/Squeeze_grad/Shape*
T0*
Tshape0
i
.gradients_1/critic/Conv1D_2/Squeeze_grad/ShapeShapecritic/Conv1D_2/BiasAdd*
T0*
out_type0
�
0gradients_1/critic/Conv1D_2/Squeeze_grad/ReshapeReshape.gradients_1/critic/Conv1D_2/Relu_grad/ReluGrad.gradients_1/critic/Conv1D_2/Squeeze_grad/Shape*
T0*
Tshape0
�
6gradients_1/critic/FullyConnected_2/MatMul_grad/MatMulMatMul6gradients_1/critic/FullyConnected_2/Relu_grad/ReluGradcritic/FullyConnected_2/W/read*
T0*
transpose_a( *
transpose_b(
�
8gradients_1/critic/FullyConnected_2/MatMul_grad/MatMul_1MatMulcritic/strided_slice_56gradients_1/critic/FullyConnected_2/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *
T0
�
2gradients_1/critic/Conv1D/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/critic/Conv1D/Squeeze_grad/Reshape*
data_formatNHWC*
T0
�
4gradients_1/critic/Conv1D_1/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients_1/critic/Conv1D_1/Squeeze_grad/Reshape*
T0*
data_formatNHWC
�
4gradients_1/critic/Conv1D_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients_1/critic/Conv1D_2/Squeeze_grad/Reshape*
T0*
data_formatNHWC
g
+gradients_1/critic/Conv1D/Conv2D_grad/ShapeShapecritic/Conv1D/ExpandDims*
T0*
out_type0
�
9gradients_1/critic/Conv1D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput+gradients_1/critic/Conv1D/Conv2D_grad/Shapecritic/Conv1D/W/read.gradients_1/critic/Conv1D/Squeeze_grad/Reshape*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
j
-gradients_1/critic/Conv1D/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"         �   
�
:gradients_1/critic/Conv1D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercritic/Conv1D/ExpandDims-gradients_1/critic/Conv1D/Conv2D_grad/Shape_1.gradients_1/critic/Conv1D/Squeeze_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
k
-gradients_1/critic/Conv1D_1/Conv2D_grad/ShapeShapecritic/Conv1D_1/ExpandDims*
T0*
out_type0
�
;gradients_1/critic/Conv1D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_1/critic/Conv1D_1/Conv2D_grad/Shapecritic/Conv1D_1/W/read0gradients_1/critic/Conv1D_1/Squeeze_grad/Reshape*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
l
/gradients_1/critic/Conv1D_1/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0
�
<gradients_1/critic/Conv1D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercritic/Conv1D_1/ExpandDims/gradients_1/critic/Conv1D_1/Conv2D_grad/Shape_10gradients_1/critic/Conv1D_1/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
k
-gradients_1/critic/Conv1D_2/Conv2D_grad/ShapeShapecritic/Conv1D_2/ExpandDims*
T0*
out_type0
�
;gradients_1/critic/Conv1D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-gradients_1/critic/Conv1D_2/Conv2D_grad/Shapecritic/Conv1D_2/W/read0gradients_1/critic/Conv1D_2/Squeeze_grad/Reshape*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
l
/gradients_1/critic/Conv1D_2/Conv2D_grad/Shape_1Const*%
valueB"         �   *
dtype0
�
<gradients_1/critic/Conv1D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltercritic/Conv1D_2/ExpandDims/gradients_1/critic/Conv1D_2/Conv2D_grad/Shape_10gradients_1/critic/Conv1D_2/Squeeze_grad/Reshape*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
�
1critic/FullyConnected/W/RMSProp/Initializer/ConstConst*
valueB	�*  �?**
_class 
loc:@critic/FullyConnected/W*
dtype0
�
critic/FullyConnected/W/RMSProp
VariableV2*
dtype0*
	container *
shape:	�*
shared_name **
_class 
loc:@critic/FullyConnected/W
�
&critic/FullyConnected/W/RMSProp/AssignIdentity1critic/FullyConnected/W/RMSProp/Initializer/Const*
T0**
_class 
loc:@critic/FullyConnected/W
�
$critic/FullyConnected/W/RMSProp/readIdentitycritic/FullyConnected/W/RMSProp*
T0**
_class 
loc:@critic/FullyConnected/W
�
3critic/FullyConnected/W/RMSProp_1/Initializer/ConstConst*
valueB	�*    **
_class 
loc:@critic/FullyConnected/W*
dtype0
�
!critic/FullyConnected/W/RMSProp_1
VariableV2*
shared_name **
_class 
loc:@critic/FullyConnected/W*
dtype0*
	container *
shape:	�
�
(critic/FullyConnected/W/RMSProp_1/AssignIdentity3critic/FullyConnected/W/RMSProp_1/Initializer/Const*
T0**
_class 
loc:@critic/FullyConnected/W
�
&critic/FullyConnected/W/RMSProp_1/readIdentity!critic/FullyConnected/W/RMSProp_1*
T0**
_class 
loc:@critic/FullyConnected/W
�
1critic/FullyConnected/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?**
_class 
loc:@critic/FullyConnected/b
�
critic/FullyConnected/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name **
_class 
loc:@critic/FullyConnected/b
�
&critic/FullyConnected/b/RMSProp/AssignIdentity1critic/FullyConnected/b/RMSProp/Initializer/Const*
T0**
_class 
loc:@critic/FullyConnected/b
�
$critic/FullyConnected/b/RMSProp/readIdentitycritic/FullyConnected/b/RMSProp*
T0**
_class 
loc:@critic/FullyConnected/b
�
3critic/FullyConnected/b/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB�*    **
_class 
loc:@critic/FullyConnected/b
�
!critic/FullyConnected/b/RMSProp_1
VariableV2*
shape:�*
shared_name **
_class 
loc:@critic/FullyConnected/b*
dtype0*
	container 
�
(critic/FullyConnected/b/RMSProp_1/AssignIdentity3critic/FullyConnected/b/RMSProp_1/Initializer/Const*
T0**
_class 
loc:@critic/FullyConnected/b
�
&critic/FullyConnected/b/RMSProp_1/readIdentity!critic/FullyConnected/b/RMSProp_1*
T0**
_class 
loc:@critic/FullyConnected/b
�
3critic/FullyConnected_1/W/RMSProp/Initializer/ConstConst*
dtype0*
valueB	�*  �?*,
_class"
 loc:@critic/FullyConnected_1/W
�
!critic/FullyConnected_1/W/RMSProp
VariableV2*
shared_name *,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0*
	container *
shape:	�
�
(critic/FullyConnected_1/W/RMSProp/AssignIdentity3critic/FullyConnected_1/W/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
&critic/FullyConnected_1/W/RMSProp/readIdentity!critic/FullyConnected_1/W/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
5critic/FullyConnected_1/W/RMSProp_1/Initializer/ConstConst*
valueB	�*    *,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0
�
#critic/FullyConnected_1/W/RMSProp_1
VariableV2*,
_class"
 loc:@critic/FullyConnected_1/W*
dtype0*
	container *
shape:	�*
shared_name 
�
*critic/FullyConnected_1/W/RMSProp_1/AssignIdentity5critic/FullyConnected_1/W/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
(critic/FullyConnected_1/W/RMSProp_1/readIdentity#critic/FullyConnected_1/W/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
3critic/FullyConnected_1/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*,
_class"
 loc:@critic/FullyConnected_1/b
�
!critic/FullyConnected_1/b/RMSProp
VariableV2*,
_class"
 loc:@critic/FullyConnected_1/b*
dtype0*
	container *
shape:�*
shared_name 
�
(critic/FullyConnected_1/b/RMSProp/AssignIdentity3critic/FullyConnected_1/b/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
&critic/FullyConnected_1/b/RMSProp/readIdentity!critic/FullyConnected_1/b/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
5critic/FullyConnected_1/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_1/b*
dtype0
�
#critic/FullyConnected_1/b/RMSProp_1
VariableV2*,
_class"
 loc:@critic/FullyConnected_1/b*
dtype0*
	container *
shape:�*
shared_name 
�
*critic/FullyConnected_1/b/RMSProp_1/AssignIdentity5critic/FullyConnected_1/b/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
(critic/FullyConnected_1/b/RMSProp_1/readIdentity#critic/FullyConnected_1/b/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
)critic/Conv1D/W/RMSProp/Initializer/ConstConst*
dtype0*&
valueB�*  �?*"
_class
loc:@critic/Conv1D/W
�
critic/Conv1D/W/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *"
_class
loc:@critic/Conv1D/W
�
critic/Conv1D/W/RMSProp/AssignIdentity)critic/Conv1D/W/RMSProp/Initializer/Const*
T0*"
_class
loc:@critic/Conv1D/W
n
critic/Conv1D/W/RMSProp/readIdentitycritic/Conv1D/W/RMSProp*
T0*"
_class
loc:@critic/Conv1D/W
�
+critic/Conv1D/W/RMSProp_1/Initializer/ConstConst*&
valueB�*    *"
_class
loc:@critic/Conv1D/W*
dtype0
�
critic/Conv1D/W/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *"
_class
loc:@critic/Conv1D/W
�
 critic/Conv1D/W/RMSProp_1/AssignIdentity+critic/Conv1D/W/RMSProp_1/Initializer/Const*
T0*"
_class
loc:@critic/Conv1D/W
r
critic/Conv1D/W/RMSProp_1/readIdentitycritic/Conv1D/W/RMSProp_1*
T0*"
_class
loc:@critic/Conv1D/W

)critic/Conv1D/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*"
_class
loc:@critic/Conv1D/b*
dtype0
�
critic/Conv1D/b/RMSProp
VariableV2*
shared_name *"
_class
loc:@critic/Conv1D/b*
dtype0*
	container *
shape:�
�
critic/Conv1D/b/RMSProp/AssignIdentity)critic/Conv1D/b/RMSProp/Initializer/Const*
T0*"
_class
loc:@critic/Conv1D/b
n
critic/Conv1D/b/RMSProp/readIdentitycritic/Conv1D/b/RMSProp*
T0*"
_class
loc:@critic/Conv1D/b
�
+critic/Conv1D/b/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB�*    *"
_class
loc:@critic/Conv1D/b
�
critic/Conv1D/b/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *"
_class
loc:@critic/Conv1D/b
�
 critic/Conv1D/b/RMSProp_1/AssignIdentity+critic/Conv1D/b/RMSProp_1/Initializer/Const*
T0*"
_class
loc:@critic/Conv1D/b
r
critic/Conv1D/b/RMSProp_1/readIdentitycritic/Conv1D/b/RMSProp_1*
T0*"
_class
loc:@critic/Conv1D/b
�
+critic/Conv1D_1/W/RMSProp/Initializer/ConstConst*&
valueB�*  �?*$
_class
loc:@critic/Conv1D_1/W*
dtype0
�
critic/Conv1D_1/W/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_1/W
�
 critic/Conv1D_1/W/RMSProp/AssignIdentity+critic/Conv1D_1/W/RMSProp/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_1/W
t
critic/Conv1D_1/W/RMSProp/readIdentitycritic/Conv1D_1/W/RMSProp*
T0*$
_class
loc:@critic/Conv1D_1/W
�
-critic/Conv1D_1/W/RMSProp_1/Initializer/ConstConst*&
valueB�*    *$
_class
loc:@critic/Conv1D_1/W*
dtype0
�
critic/Conv1D_1/W/RMSProp_1
VariableV2*
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_1/W*
dtype0*
	container 
�
"critic/Conv1D_1/W/RMSProp_1/AssignIdentity-critic/Conv1D_1/W/RMSProp_1/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_1/W
x
 critic/Conv1D_1/W/RMSProp_1/readIdentitycritic/Conv1D_1/W/RMSProp_1*
T0*$
_class
loc:@critic/Conv1D_1/W
�
+critic/Conv1D_1/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*$
_class
loc:@critic/Conv1D_1/b
�
critic/Conv1D_1/b/RMSProp
VariableV2*$
_class
loc:@critic/Conv1D_1/b*
dtype0*
	container *
shape:�*
shared_name 
�
 critic/Conv1D_1/b/RMSProp/AssignIdentity+critic/Conv1D_1/b/RMSProp/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_1/b
t
critic/Conv1D_1/b/RMSProp/readIdentitycritic/Conv1D_1/b/RMSProp*
T0*$
_class
loc:@critic/Conv1D_1/b
�
-critic/Conv1D_1/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *$
_class
loc:@critic/Conv1D_1/b*
dtype0
�
critic/Conv1D_1/b/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_1/b
�
"critic/Conv1D_1/b/RMSProp_1/AssignIdentity-critic/Conv1D_1/b/RMSProp_1/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_1/b
x
 critic/Conv1D_1/b/RMSProp_1/readIdentitycritic/Conv1D_1/b/RMSProp_1*
T0*$
_class
loc:@critic/Conv1D_1/b
�
+critic/Conv1D_2/W/RMSProp/Initializer/ConstConst*
dtype0*&
valueB�*  �?*$
_class
loc:@critic/Conv1D_2/W
�
critic/Conv1D_2/W/RMSProp
VariableV2*
shared_name *$
_class
loc:@critic/Conv1D_2/W*
dtype0*
	container *
shape:�
�
 critic/Conv1D_2/W/RMSProp/AssignIdentity+critic/Conv1D_2/W/RMSProp/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_2/W
t
critic/Conv1D_2/W/RMSProp/readIdentitycritic/Conv1D_2/W/RMSProp*
T0*$
_class
loc:@critic/Conv1D_2/W
�
-critic/Conv1D_2/W/RMSProp_1/Initializer/ConstConst*
dtype0*&
valueB�*    *$
_class
loc:@critic/Conv1D_2/W
�
critic/Conv1D_2/W/RMSProp_1
VariableV2*$
_class
loc:@critic/Conv1D_2/W*
dtype0*
	container *
shape:�*
shared_name 
�
"critic/Conv1D_2/W/RMSProp_1/AssignIdentity-critic/Conv1D_2/W/RMSProp_1/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_2/W
x
 critic/Conv1D_2/W/RMSProp_1/readIdentitycritic/Conv1D_2/W/RMSProp_1*
T0*$
_class
loc:@critic/Conv1D_2/W
�
+critic/Conv1D_2/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*$
_class
loc:@critic/Conv1D_2/b*
dtype0
�
critic/Conv1D_2/b/RMSProp
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_2/b
�
 critic/Conv1D_2/b/RMSProp/AssignIdentity+critic/Conv1D_2/b/RMSProp/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_2/b
t
critic/Conv1D_2/b/RMSProp/readIdentitycritic/Conv1D_2/b/RMSProp*
T0*$
_class
loc:@critic/Conv1D_2/b
�
-critic/Conv1D_2/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *$
_class
loc:@critic/Conv1D_2/b*
dtype0
�
critic/Conv1D_2/b/RMSProp_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *$
_class
loc:@critic/Conv1D_2/b
�
"critic/Conv1D_2/b/RMSProp_1/AssignIdentity-critic/Conv1D_2/b/RMSProp_1/Initializer/Const*
T0*$
_class
loc:@critic/Conv1D_2/b
x
 critic/Conv1D_2/b/RMSProp_1/readIdentitycritic/Conv1D_2/b/RMSProp_1*
T0*$
_class
loc:@critic/Conv1D_2/b
�
3critic/FullyConnected_2/W/RMSProp/Initializer/ConstConst*
dtype0*
valueB	�*  �?*,
_class"
 loc:@critic/FullyConnected_2/W
�
!critic/FullyConnected_2/W/RMSProp
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_2/W
�
(critic/FullyConnected_2/W/RMSProp/AssignIdentity3critic/FullyConnected_2/W/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
&critic/FullyConnected_2/W/RMSProp/readIdentity!critic/FullyConnected_2/W/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
5critic/FullyConnected_2/W/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB	�*    *,
_class"
 loc:@critic/FullyConnected_2/W
�
#critic/FullyConnected_2/W/RMSProp_1
VariableV2*
dtype0*
	container *
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_2/W
�
*critic/FullyConnected_2/W/RMSProp_1/AssignIdentity5critic/FullyConnected_2/W/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
(critic/FullyConnected_2/W/RMSProp_1/readIdentity#critic/FullyConnected_2/W/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
3critic/FullyConnected_2/b/RMSProp/Initializer/ConstConst*
valueB�*  �?*,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0
�
!critic/FullyConnected_2/b/RMSProp
VariableV2*,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0*
	container *
shape:�*
shared_name 
�
(critic/FullyConnected_2/b/RMSProp/AssignIdentity3critic/FullyConnected_2/b/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
�
&critic/FullyConnected_2/b/RMSProp/readIdentity!critic/FullyConnected_2/b/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
�
5critic/FullyConnected_2/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0
�
#critic/FullyConnected_2/b/RMSProp_1
VariableV2*,
_class"
 loc:@critic/FullyConnected_2/b*
dtype0*
	container *
shape:�*
shared_name 
�
*critic/FullyConnected_2/b/RMSProp_1/AssignIdentity5critic/FullyConnected_2/b/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
�
(critic/FullyConnected_2/b/RMSProp_1/readIdentity#critic/FullyConnected_2/b/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
�
3critic/FullyConnected_3/W/RMSProp/Initializer/ConstConst*
valueB
��*  �?*,
_class"
 loc:@critic/FullyConnected_3/W*
dtype0
�
!critic/FullyConnected_3/W/RMSProp
VariableV2*
dtype0*
	container *
shape:
��*
shared_name *,
_class"
 loc:@critic/FullyConnected_3/W
�
(critic/FullyConnected_3/W/RMSProp/AssignIdentity3critic/FullyConnected_3/W/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
&critic/FullyConnected_3/W/RMSProp/readIdentity!critic/FullyConnected_3/W/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
5critic/FullyConnected_3/W/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB
��*    *,
_class"
 loc:@critic/FullyConnected_3/W
�
#critic/FullyConnected_3/W/RMSProp_1
VariableV2*
shape:
��*
shared_name *,
_class"
 loc:@critic/FullyConnected_3/W*
dtype0*
	container 
�
*critic/FullyConnected_3/W/RMSProp_1/AssignIdentity5critic/FullyConnected_3/W/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
(critic/FullyConnected_3/W/RMSProp_1/readIdentity#critic/FullyConnected_3/W/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
3critic/FullyConnected_3/b/RMSProp/Initializer/ConstConst*
dtype0*
valueB�*  �?*,
_class"
 loc:@critic/FullyConnected_3/b
�
!critic/FullyConnected_3/b/RMSProp
VariableV2*,
_class"
 loc:@critic/FullyConnected_3/b*
dtype0*
	container *
shape:�*
shared_name 
�
(critic/FullyConnected_3/b/RMSProp/AssignIdentity3critic/FullyConnected_3/b/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
&critic/FullyConnected_3/b/RMSProp/readIdentity!critic/FullyConnected_3/b/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
5critic/FullyConnected_3/b/RMSProp_1/Initializer/ConstConst*
valueB�*    *,
_class"
 loc:@critic/FullyConnected_3/b*
dtype0
�
#critic/FullyConnected_3/b/RMSProp_1
VariableV2*,
_class"
 loc:@critic/FullyConnected_3/b*
dtype0*
	container *
shape:�*
shared_name 
�
*critic/FullyConnected_3/b/RMSProp_1/AssignIdentity5critic/FullyConnected_3/b/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
(critic/FullyConnected_3/b/RMSProp_1/readIdentity#critic/FullyConnected_3/b/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
3critic/FullyConnected_4/W/RMSProp/Initializer/ConstConst*
dtype0*
valueB	�*  �?*,
_class"
 loc:@critic/FullyConnected_4/W
�
!critic/FullyConnected_4/W/RMSProp
VariableV2*
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/W*
dtype0*
	container 
�
(critic/FullyConnected_4/W/RMSProp/AssignIdentity3critic/FullyConnected_4/W/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
&critic/FullyConnected_4/W/RMSProp/readIdentity!critic/FullyConnected_4/W/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
5critic/FullyConnected_4/W/RMSProp_1/Initializer/ConstConst*
dtype0*
valueB	�*    *,
_class"
 loc:@critic/FullyConnected_4/W
�
#critic/FullyConnected_4/W/RMSProp_1
VariableV2*
shape:	�*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/W*
dtype0*
	container 
�
*critic/FullyConnected_4/W/RMSProp_1/AssignIdentity5critic/FullyConnected_4/W/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
(critic/FullyConnected_4/W/RMSProp_1/readIdentity#critic/FullyConnected_4/W/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
3critic/FullyConnected_4/b/RMSProp/Initializer/ConstConst*
valueB*  �?*,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0
�
!critic/FullyConnected_4/b/RMSProp
VariableV2*
shape:*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0*
	container 
�
(critic/FullyConnected_4/b/RMSProp/AssignIdentity3critic/FullyConnected_4/b/RMSProp/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
�
&critic/FullyConnected_4/b/RMSProp/readIdentity!critic/FullyConnected_4/b/RMSProp*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
�
5critic/FullyConnected_4/b/RMSProp_1/Initializer/ConstConst*
valueB*    *,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0
�
#critic/FullyConnected_4/b/RMSProp_1
VariableV2*
shape:*
shared_name *,
_class"
 loc:@critic/FullyConnected_4/b*
dtype0*
	container 
�
*critic/FullyConnected_4/b/RMSProp_1/AssignIdentity5critic/FullyConnected_4/b/RMSProp_1/Initializer/Const*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
�
(critic/FullyConnected_4/b/RMSProp_1/readIdentity#critic/FullyConnected_4/b/RMSProp_1*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
D
RMSProp_1/learning_rateConst*
valueB
 *��8*
dtype0
<
RMSProp_1/decayConst*
dtype0*
valueB
 *fff?
?
RMSProp_1/momentumConst*
valueB
 *    *
dtype0
>
RMSProp_1/epsilonConst*
valueB
 *���.*
dtype0
�
5RMSProp_1/update_critic/FullyConnected/W/ApplyRMSPropApplyRMSPropcritic/FullyConnected/Wcritic/FullyConnected/W/RMSProp!critic/FullyConnected/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon6gradients_1/critic/FullyConnected/MatMul_grad/MatMul_1*
T0**
_class 
loc:@critic/FullyConnected/W*
use_locking( 
�
5RMSProp_1/update_critic/FullyConnected/b/ApplyRMSPropApplyRMSPropcritic/FullyConnected/bcritic/FullyConnected/b/RMSProp!critic/FullyConnected/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon:gradients_1/critic/FullyConnected/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0**
_class 
loc:@critic/FullyConnected/b
�
7RMSProp_1/update_critic/FullyConnected_1/W/ApplyRMSPropApplyRMSPropcritic/FullyConnected_1/W!critic/FullyConnected_1/W/RMSProp#critic/FullyConnected_1/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon8gradients_1/critic/FullyConnected_1/MatMul_grad/MatMul_1*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_1/W
�
7RMSProp_1/update_critic/FullyConnected_1/b/ApplyRMSPropApplyRMSPropcritic/FullyConnected_1/b!critic/FullyConnected_1/b/RMSProp#critic/FullyConnected_1/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/FullyConnected_1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_1/b
�
-RMSProp_1/update_critic/Conv1D/W/ApplyRMSPropApplyRMSPropcritic/Conv1D/Wcritic/Conv1D/W/RMSPropcritic/Conv1D/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon:gradients_1/critic/Conv1D/Conv2D_grad/Conv2DBackpropFilter*
T0*"
_class
loc:@critic/Conv1D/W*
use_locking( 
�
-RMSProp_1/update_critic/Conv1D/b/ApplyRMSPropApplyRMSPropcritic/Conv1D/bcritic/Conv1D/b/RMSPropcritic/Conv1D/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon2gradients_1/critic/Conv1D/BiasAdd_grad/BiasAddGrad*
T0*"
_class
loc:@critic/Conv1D/b*
use_locking( 
�
/RMSProp_1/update_critic/Conv1D_1/W/ApplyRMSPropApplyRMSPropcritic/Conv1D_1/Wcritic/Conv1D_1/W/RMSPropcritic/Conv1D_1/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/Conv1D_1/Conv2D_grad/Conv2DBackpropFilter*
T0*$
_class
loc:@critic/Conv1D_1/W*
use_locking( 
�
/RMSProp_1/update_critic/Conv1D_1/b/ApplyRMSPropApplyRMSPropcritic/Conv1D_1/bcritic/Conv1D_1/b/RMSPropcritic/Conv1D_1/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon4gradients_1/critic/Conv1D_1/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*$
_class
loc:@critic/Conv1D_1/b
�
/RMSProp_1/update_critic/Conv1D_2/W/ApplyRMSPropApplyRMSPropcritic/Conv1D_2/Wcritic/Conv1D_2/W/RMSPropcritic/Conv1D_2/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/Conv1D_2/Conv2D_grad/Conv2DBackpropFilter*
use_locking( *
T0*$
_class
loc:@critic/Conv1D_2/W
�
/RMSProp_1/update_critic/Conv1D_2/b/ApplyRMSPropApplyRMSPropcritic/Conv1D_2/bcritic/Conv1D_2/b/RMSPropcritic/Conv1D_2/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon4gradients_1/critic/Conv1D_2/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*$
_class
loc:@critic/Conv1D_2/b
�
7RMSProp_1/update_critic/FullyConnected_2/W/ApplyRMSPropApplyRMSPropcritic/FullyConnected_2/W!critic/FullyConnected_2/W/RMSProp#critic/FullyConnected_2/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon8gradients_1/critic/FullyConnected_2/MatMul_grad/MatMul_1*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_2/W
�
7RMSProp_1/update_critic/FullyConnected_2/b/ApplyRMSPropApplyRMSPropcritic/FullyConnected_2/b!critic/FullyConnected_2/b/RMSProp#critic/FullyConnected_2/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/FullyConnected_2/BiasAdd_grad/BiasAddGrad*
T0*,
_class"
 loc:@critic/FullyConnected_2/b*
use_locking( 
�
7RMSProp_1/update_critic/FullyConnected_3/W/ApplyRMSPropApplyRMSPropcritic/FullyConnected_3/W!critic/FullyConnected_3/W/RMSProp#critic/FullyConnected_3/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon8gradients_1/critic/FullyConnected_3/MatMul_grad/MatMul_1*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_3/W
�
7RMSProp_1/update_critic/FullyConnected_3/b/ApplyRMSPropApplyRMSPropcritic/FullyConnected_3/b!critic/FullyConnected_3/b/RMSProp#critic/FullyConnected_3/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/FullyConnected_3/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_3/b
�
7RMSProp_1/update_critic/FullyConnected_4/W/ApplyRMSPropApplyRMSPropcritic/FullyConnected_4/W!critic/FullyConnected_4/W/RMSProp#critic/FullyConnected_4/W/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon8gradients_1/critic/FullyConnected_4/MatMul_grad/MatMul_1*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_4/W
�
7RMSProp_1/update_critic/FullyConnected_4/b/ApplyRMSPropApplyRMSPropcritic/FullyConnected_4/b!critic/FullyConnected_4/b/RMSProp#critic/FullyConnected_4/b/RMSProp_1RMSProp_1/learning_rateRMSProp_1/decayRMSProp_1/momentumRMSProp_1/epsilon<gradients_1/critic/FullyConnected_4/BiasAdd_grad/BiasAddGrad*
use_locking( *
T0*,
_class"
 loc:@critic/FullyConnected_4/b
�
	RMSProp_1NoOp.^RMSProp_1/update_critic/Conv1D/W/ApplyRMSProp.^RMSProp_1/update_critic/Conv1D/b/ApplyRMSProp0^RMSProp_1/update_critic/Conv1D_1/W/ApplyRMSProp0^RMSProp_1/update_critic/Conv1D_1/b/ApplyRMSProp0^RMSProp_1/update_critic/Conv1D_2/W/ApplyRMSProp0^RMSProp_1/update_critic/Conv1D_2/b/ApplyRMSProp6^RMSProp_1/update_critic/FullyConnected/W/ApplyRMSProp6^RMSProp_1/update_critic/FullyConnected/b/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_1/W/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_1/b/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_2/W/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_2/b/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_3/W/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_3/b/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_4/W/ApplyRMSProp8^RMSProp_1/update_critic/FullyConnected_4/b/ApplyRMSProp
C
Variable/initial_valueConst*
dtype0*
valueB
 *    
T
Variable
VariableV2*
dtype0*
	container *
shape: *
shared_name 
Y
Variable/AssignIdentityVariable/initial_value*
T0*
_class
loc:@Variable
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
<
TD_loss/tagsConst*
valueB BTD_loss*
dtype0
>
TD_lossScalarSummaryTD_loss/tagsVariable/read*
T0
E
Variable_1/initial_valueConst*
dtype0*
valueB
 *    
V

Variable_1
VariableV2*
dtype0*
	container *
shape: *
shared_name 
_
Variable_1/AssignIdentityVariable_1/initial_value*
T0*
_class
loc:@Variable_1
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
N
Eps_total_reward/tagsConst*!
valueB BEps_total_reward*
dtype0
R
Eps_total_rewardScalarSummaryEps_total_reward/tagsVariable_1/read*
T0
E
Variable_2/initial_valueConst*
valueB
 *    *
dtype0
V

Variable_2
VariableV2*
shape: *
shared_name *
dtype0*
	container 
_
Variable_2/AssignIdentityVariable_2/initial_value*
T0*
_class
loc:@Variable_2
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
D
Avg_entropy/tagsConst*
valueB BAvg_entropy*
dtype0
H
Avg_entropyScalarSummaryAvg_entropy/tagsVariable_2/read*
T0
S
Merge/MergeSummaryMergeSummaryTD_lossEps_total_rewardAvg_entropy*
N
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^actor/Conv1D/W/Assign^actor/Conv1D/W/RMSProp/Assign ^actor/Conv1D/W/RMSProp_1/Assign^actor/Conv1D/b/Assign^actor/Conv1D/b/RMSProp/Assign ^actor/Conv1D/b/RMSProp_1/Assign^actor/Conv1D_1/W/Assign ^actor/Conv1D_1/W/RMSProp/Assign"^actor/Conv1D_1/W/RMSProp_1/Assign^actor/Conv1D_1/b/Assign ^actor/Conv1D_1/b/RMSProp/Assign"^actor/Conv1D_1/b/RMSProp_1/Assign^actor/Conv1D_2/W/Assign ^actor/Conv1D_2/W/RMSProp/Assign"^actor/Conv1D_2/W/RMSProp_1/Assign^actor/Conv1D_2/b/Assign ^actor/Conv1D_2/b/RMSProp/Assign"^actor/Conv1D_2/b/RMSProp_1/Assign^actor/FullyConnected/W/Assign&^actor/FullyConnected/W/RMSProp/Assign(^actor/FullyConnected/W/RMSProp_1/Assign^actor/FullyConnected/b/Assign&^actor/FullyConnected/b/RMSProp/Assign(^actor/FullyConnected/b/RMSProp_1/Assign ^actor/FullyConnected_1/W/Assign(^actor/FullyConnected_1/W/RMSProp/Assign*^actor/FullyConnected_1/W/RMSProp_1/Assign ^actor/FullyConnected_1/b/Assign(^actor/FullyConnected_1/b/RMSProp/Assign*^actor/FullyConnected_1/b/RMSProp_1/Assign ^actor/FullyConnected_2/W/Assign(^actor/FullyConnected_2/W/RMSProp/Assign*^actor/FullyConnected_2/W/RMSProp_1/Assign ^actor/FullyConnected_2/b/Assign(^actor/FullyConnected_2/b/RMSProp/Assign*^actor/FullyConnected_2/b/RMSProp_1/Assign ^actor/FullyConnected_3/W/Assign(^actor/FullyConnected_3/W/RMSProp/Assign*^actor/FullyConnected_3/W/RMSProp_1/Assign ^actor/FullyConnected_3/b/Assign(^actor/FullyConnected_3/b/RMSProp/Assign*^actor/FullyConnected_3/b/RMSProp_1/Assign ^actor/FullyConnected_4/W/Assign(^actor/FullyConnected_4/W/RMSProp/Assign*^actor/FullyConnected_4/W/RMSProp_1/Assign ^actor/FullyConnected_4/b/Assign(^actor/FullyConnected_4/b/RMSProp/Assign*^actor/FullyConnected_4/b/RMSProp_1/Assign^critic/Conv1D/W/Assign^critic/Conv1D/W/RMSProp/Assign!^critic/Conv1D/W/RMSProp_1/Assign^critic/Conv1D/b/Assign^critic/Conv1D/b/RMSProp/Assign!^critic/Conv1D/b/RMSProp_1/Assign^critic/Conv1D_1/W/Assign!^critic/Conv1D_1/W/RMSProp/Assign#^critic/Conv1D_1/W/RMSProp_1/Assign^critic/Conv1D_1/b/Assign!^critic/Conv1D_1/b/RMSProp/Assign#^critic/Conv1D_1/b/RMSProp_1/Assign^critic/Conv1D_2/W/Assign!^critic/Conv1D_2/W/RMSProp/Assign#^critic/Conv1D_2/W/RMSProp_1/Assign^critic/Conv1D_2/b/Assign!^critic/Conv1D_2/b/RMSProp/Assign#^critic/Conv1D_2/b/RMSProp_1/Assign^critic/FullyConnected/W/Assign'^critic/FullyConnected/W/RMSProp/Assign)^critic/FullyConnected/W/RMSProp_1/Assign^critic/FullyConnected/b/Assign'^critic/FullyConnected/b/RMSProp/Assign)^critic/FullyConnected/b/RMSProp_1/Assign!^critic/FullyConnected_1/W/Assign)^critic/FullyConnected_1/W/RMSProp/Assign+^critic/FullyConnected_1/W/RMSProp_1/Assign!^critic/FullyConnected_1/b/Assign)^critic/FullyConnected_1/b/RMSProp/Assign+^critic/FullyConnected_1/b/RMSProp_1/Assign!^critic/FullyConnected_2/W/Assign)^critic/FullyConnected_2/W/RMSProp/Assign+^critic/FullyConnected_2/W/RMSProp_1/Assign!^critic/FullyConnected_2/b/Assign)^critic/FullyConnected_2/b/RMSProp/Assign+^critic/FullyConnected_2/b/RMSProp_1/Assign!^critic/FullyConnected_3/W/Assign)^critic/FullyConnected_3/W/RMSProp/Assign+^critic/FullyConnected_3/W/RMSProp_1/Assign!^critic/FullyConnected_3/b/Assign)^critic/FullyConnected_3/b/RMSProp/Assign+^critic/FullyConnected_3/b/RMSProp_1/Assign!^critic/FullyConnected_4/W/Assign)^critic/FullyConnected_4/W/RMSProp/Assign+^critic/FullyConnected_4/W/RMSProp_1/Assign!^critic/FullyConnected_4/b/Assign)^critic/FullyConnected_4/b/RMSProp/Assign+^critic/FullyConnected_4/b/RMSProp_1/Assign^is_training/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*�
value�B�dBVariableB
Variable_1B
Variable_2Bactor/Conv1D/WBactor/Conv1D/W/RMSPropBactor/Conv1D/W/RMSProp_1Bactor/Conv1D/bBactor/Conv1D/b/RMSPropBactor/Conv1D/b/RMSProp_1Bactor/Conv1D_1/WBactor/Conv1D_1/W/RMSPropBactor/Conv1D_1/W/RMSProp_1Bactor/Conv1D_1/bBactor/Conv1D_1/b/RMSPropBactor/Conv1D_1/b/RMSProp_1Bactor/Conv1D_2/WBactor/Conv1D_2/W/RMSPropBactor/Conv1D_2/W/RMSProp_1Bactor/Conv1D_2/bBactor/Conv1D_2/b/RMSPropBactor/Conv1D_2/b/RMSProp_1Bactor/FullyConnected/WBactor/FullyConnected/W/RMSPropB actor/FullyConnected/W/RMSProp_1Bactor/FullyConnected/bBactor/FullyConnected/b/RMSPropB actor/FullyConnected/b/RMSProp_1Bactor/FullyConnected_1/WB actor/FullyConnected_1/W/RMSPropB"actor/FullyConnected_1/W/RMSProp_1Bactor/FullyConnected_1/bB actor/FullyConnected_1/b/RMSPropB"actor/FullyConnected_1/b/RMSProp_1Bactor/FullyConnected_2/WB actor/FullyConnected_2/W/RMSPropB"actor/FullyConnected_2/W/RMSProp_1Bactor/FullyConnected_2/bB actor/FullyConnected_2/b/RMSPropB"actor/FullyConnected_2/b/RMSProp_1Bactor/FullyConnected_3/WB actor/FullyConnected_3/W/RMSPropB"actor/FullyConnected_3/W/RMSProp_1Bactor/FullyConnected_3/bB actor/FullyConnected_3/b/RMSPropB"actor/FullyConnected_3/b/RMSProp_1Bactor/FullyConnected_4/WB actor/FullyConnected_4/W/RMSPropB"actor/FullyConnected_4/W/RMSProp_1Bactor/FullyConnected_4/bB actor/FullyConnected_4/b/RMSPropB"actor/FullyConnected_4/b/RMSProp_1Bcritic/Conv1D/WBcritic/Conv1D/W/RMSPropBcritic/Conv1D/W/RMSProp_1Bcritic/Conv1D/bBcritic/Conv1D/b/RMSPropBcritic/Conv1D/b/RMSProp_1Bcritic/Conv1D_1/WBcritic/Conv1D_1/W/RMSPropBcritic/Conv1D_1/W/RMSProp_1Bcritic/Conv1D_1/bBcritic/Conv1D_1/b/RMSPropBcritic/Conv1D_1/b/RMSProp_1Bcritic/Conv1D_2/WBcritic/Conv1D_2/W/RMSPropBcritic/Conv1D_2/W/RMSProp_1Bcritic/Conv1D_2/bBcritic/Conv1D_2/b/RMSPropBcritic/Conv1D_2/b/RMSProp_1Bcritic/FullyConnected/WBcritic/FullyConnected/W/RMSPropB!critic/FullyConnected/W/RMSProp_1Bcritic/FullyConnected/bBcritic/FullyConnected/b/RMSPropB!critic/FullyConnected/b/RMSProp_1Bcritic/FullyConnected_1/WB!critic/FullyConnected_1/W/RMSPropB#critic/FullyConnected_1/W/RMSProp_1Bcritic/FullyConnected_1/bB!critic/FullyConnected_1/b/RMSPropB#critic/FullyConnected_1/b/RMSProp_1Bcritic/FullyConnected_2/WB!critic/FullyConnected_2/W/RMSPropB#critic/FullyConnected_2/W/RMSProp_1Bcritic/FullyConnected_2/bB!critic/FullyConnected_2/b/RMSPropB#critic/FullyConnected_2/b/RMSProp_1Bcritic/FullyConnected_3/WB!critic/FullyConnected_3/W/RMSPropB#critic/FullyConnected_3/W/RMSProp_1Bcritic/FullyConnected_3/bB!critic/FullyConnected_3/b/RMSPropB#critic/FullyConnected_3/b/RMSProp_1Bcritic/FullyConnected_4/WB!critic/FullyConnected_4/W/RMSPropB#critic/FullyConnected_4/W/RMSProp_1Bcritic/FullyConnected_4/bB!critic/FullyConnected_4/b/RMSPropB#critic/FullyConnected_4/b/RMSProp_1Bis_training*
dtype0
�
save/SaveV2/shape_and_slicesConst*�
value�B�dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2actor/Conv1D/Wactor/Conv1D/W/RMSPropactor/Conv1D/W/RMSProp_1actor/Conv1D/bactor/Conv1D/b/RMSPropactor/Conv1D/b/RMSProp_1actor/Conv1D_1/Wactor/Conv1D_1/W/RMSPropactor/Conv1D_1/W/RMSProp_1actor/Conv1D_1/bactor/Conv1D_1/b/RMSPropactor/Conv1D_1/b/RMSProp_1actor/Conv1D_2/Wactor/Conv1D_2/W/RMSPropactor/Conv1D_2/W/RMSProp_1actor/Conv1D_2/bactor/Conv1D_2/b/RMSPropactor/Conv1D_2/b/RMSProp_1actor/FullyConnected/Wactor/FullyConnected/W/RMSProp actor/FullyConnected/W/RMSProp_1actor/FullyConnected/bactor/FullyConnected/b/RMSProp actor/FullyConnected/b/RMSProp_1actor/FullyConnected_1/W actor/FullyConnected_1/W/RMSProp"actor/FullyConnected_1/W/RMSProp_1actor/FullyConnected_1/b actor/FullyConnected_1/b/RMSProp"actor/FullyConnected_1/b/RMSProp_1actor/FullyConnected_2/W actor/FullyConnected_2/W/RMSProp"actor/FullyConnected_2/W/RMSProp_1actor/FullyConnected_2/b actor/FullyConnected_2/b/RMSProp"actor/FullyConnected_2/b/RMSProp_1actor/FullyConnected_3/W actor/FullyConnected_3/W/RMSProp"actor/FullyConnected_3/W/RMSProp_1actor/FullyConnected_3/b actor/FullyConnected_3/b/RMSProp"actor/FullyConnected_3/b/RMSProp_1actor/FullyConnected_4/W actor/FullyConnected_4/W/RMSProp"actor/FullyConnected_4/W/RMSProp_1actor/FullyConnected_4/b actor/FullyConnected_4/b/RMSProp"actor/FullyConnected_4/b/RMSProp_1critic/Conv1D/Wcritic/Conv1D/W/RMSPropcritic/Conv1D/W/RMSProp_1critic/Conv1D/bcritic/Conv1D/b/RMSPropcritic/Conv1D/b/RMSProp_1critic/Conv1D_1/Wcritic/Conv1D_1/W/RMSPropcritic/Conv1D_1/W/RMSProp_1critic/Conv1D_1/bcritic/Conv1D_1/b/RMSPropcritic/Conv1D_1/b/RMSProp_1critic/Conv1D_2/Wcritic/Conv1D_2/W/RMSPropcritic/Conv1D_2/W/RMSProp_1critic/Conv1D_2/bcritic/Conv1D_2/b/RMSPropcritic/Conv1D_2/b/RMSProp_1critic/FullyConnected/Wcritic/FullyConnected/W/RMSProp!critic/FullyConnected/W/RMSProp_1critic/FullyConnected/bcritic/FullyConnected/b/RMSProp!critic/FullyConnected/b/RMSProp_1critic/FullyConnected_1/W!critic/FullyConnected_1/W/RMSProp#critic/FullyConnected_1/W/RMSProp_1critic/FullyConnected_1/b!critic/FullyConnected_1/b/RMSProp#critic/FullyConnected_1/b/RMSProp_1critic/FullyConnected_2/W!critic/FullyConnected_2/W/RMSProp#critic/FullyConnected_2/W/RMSProp_1critic/FullyConnected_2/b!critic/FullyConnected_2/b/RMSProp#critic/FullyConnected_2/b/RMSProp_1critic/FullyConnected_3/W!critic/FullyConnected_3/W/RMSProp#critic/FullyConnected_3/W/RMSProp_1critic/FullyConnected_3/b!critic/FullyConnected_3/b/RMSProp#critic/FullyConnected_3/b/RMSProp_1critic/FullyConnected_4/W!critic/FullyConnected_4/W/RMSProp#critic/FullyConnected_4/W/RMSProp_1critic/FullyConnected_4/b!critic/FullyConnected_4/b/RMSProp#critic/FullyConnected_4/b/RMSProp_1is_training*r
dtypesh
f2d

e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
P
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0
L
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
M
save/AssignIdentitysave/RestoreV2*
T0*
_class
loc:@Variable
T
save/RestoreV2_1/tensor_namesConst*
valueBB
Variable_1*
dtype0
N
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2
S
save/Assign_1Identitysave/RestoreV2_1*
T0*
_class
loc:@Variable_1
T
save/RestoreV2_2/tensor_namesConst*
valueBB
Variable_2*
dtype0
N
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2
S
save/Assign_2Identitysave/RestoreV2_2*
T0*
_class
loc:@Variable_2
X
save/RestoreV2_3/tensor_namesConst*#
valueBBactor/Conv1D/W*
dtype0
N
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2
W
save/Assign_3Identitysave/RestoreV2_3*
T0*!
_class
loc:@actor/Conv1D/W
`
save/RestoreV2_4/tensor_namesConst*+
value"B Bactor/Conv1D/W/RMSProp*
dtype0
N
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
W
save/Assign_4Identitysave/RestoreV2_4*
T0*!
_class
loc:@actor/Conv1D/W
b
save/RestoreV2_5/tensor_namesConst*-
value$B"Bactor/Conv1D/W/RMSProp_1*
dtype0
N
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2
W
save/Assign_5Identitysave/RestoreV2_5*
T0*!
_class
loc:@actor/Conv1D/W
X
save/RestoreV2_6/tensor_namesConst*#
valueBBactor/Conv1D/b*
dtype0
N
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2
W
save/Assign_6Identitysave/RestoreV2_6*
T0*!
_class
loc:@actor/Conv1D/b
`
save/RestoreV2_7/tensor_namesConst*+
value"B Bactor/Conv1D/b/RMSProp*
dtype0
N
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2
W
save/Assign_7Identitysave/RestoreV2_7*
T0*!
_class
loc:@actor/Conv1D/b
b
save/RestoreV2_8/tensor_namesConst*-
value$B"Bactor/Conv1D/b/RMSProp_1*
dtype0
N
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
W
save/Assign_8Identitysave/RestoreV2_8*
T0*!
_class
loc:@actor/Conv1D/b
Z
save/RestoreV2_9/tensor_namesConst*%
valueBBactor/Conv1D_1/W*
dtype0
N
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
Y
save/Assign_9Identitysave/RestoreV2_9*
T0*#
_class
loc:@actor/Conv1D_1/W
c
save/RestoreV2_10/tensor_namesConst*-
value$B"Bactor/Conv1D_1/W/RMSProp*
dtype0
O
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2
[
save/Assign_10Identitysave/RestoreV2_10*
T0*#
_class
loc:@actor/Conv1D_1/W
e
save/RestoreV2_11/tensor_namesConst*/
value&B$Bactor/Conv1D_1/W/RMSProp_1*
dtype0
O
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2
[
save/Assign_11Identitysave/RestoreV2_11*
T0*#
_class
loc:@actor/Conv1D_1/W
[
save/RestoreV2_12/tensor_namesConst*%
valueBBactor/Conv1D_1/b*
dtype0
O
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2
[
save/Assign_12Identitysave/RestoreV2_12*
T0*#
_class
loc:@actor/Conv1D_1/b
c
save/RestoreV2_13/tensor_namesConst*-
value$B"Bactor/Conv1D_1/b/RMSProp*
dtype0
O
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2
[
save/Assign_13Identitysave/RestoreV2_13*
T0*#
_class
loc:@actor/Conv1D_1/b
e
save/RestoreV2_14/tensor_namesConst*/
value&B$Bactor/Conv1D_1/b/RMSProp_1*
dtype0
O
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2
[
save/Assign_14Identitysave/RestoreV2_14*
T0*#
_class
loc:@actor/Conv1D_1/b
[
save/RestoreV2_15/tensor_namesConst*
dtype0*%
valueBBactor/Conv1D_2/W
O
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2
[
save/Assign_15Identitysave/RestoreV2_15*
T0*#
_class
loc:@actor/Conv1D_2/W
c
save/RestoreV2_16/tensor_namesConst*-
value$B"Bactor/Conv1D_2/W/RMSProp*
dtype0
O
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2
[
save/Assign_16Identitysave/RestoreV2_16*
T0*#
_class
loc:@actor/Conv1D_2/W
e
save/RestoreV2_17/tensor_namesConst*
dtype0*/
value&B$Bactor/Conv1D_2/W/RMSProp_1
O
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2
[
save/Assign_17Identitysave/RestoreV2_17*
T0*#
_class
loc:@actor/Conv1D_2/W
[
save/RestoreV2_18/tensor_namesConst*%
valueBBactor/Conv1D_2/b*
dtype0
O
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2
[
save/Assign_18Identitysave/RestoreV2_18*
T0*#
_class
loc:@actor/Conv1D_2/b
c
save/RestoreV2_19/tensor_namesConst*-
value$B"Bactor/Conv1D_2/b/RMSProp*
dtype0
O
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2
[
save/Assign_19Identitysave/RestoreV2_19*
T0*#
_class
loc:@actor/Conv1D_2/b
e
save/RestoreV2_20/tensor_namesConst*
dtype0*/
value&B$Bactor/Conv1D_2/b/RMSProp_1
O
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2
[
save/Assign_20Identitysave/RestoreV2_20*
T0*#
_class
loc:@actor/Conv1D_2/b
a
save/RestoreV2_21/tensor_namesConst*+
value"B Bactor/FullyConnected/W*
dtype0
O
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2
a
save/Assign_21Identitysave/RestoreV2_21*
T0*)
_class
loc:@actor/FullyConnected/W
i
save/RestoreV2_22/tensor_namesConst*3
value*B(Bactor/FullyConnected/W/RMSProp*
dtype0
O
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2
a
save/Assign_22Identitysave/RestoreV2_22*
T0*)
_class
loc:@actor/FullyConnected/W
k
save/RestoreV2_23/tensor_namesConst*5
value,B*B actor/FullyConnected/W/RMSProp_1*
dtype0
O
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2
a
save/Assign_23Identitysave/RestoreV2_23*
T0*)
_class
loc:@actor/FullyConnected/W
a
save/RestoreV2_24/tensor_namesConst*+
value"B Bactor/FullyConnected/b*
dtype0
O
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2
a
save/Assign_24Identitysave/RestoreV2_24*
T0*)
_class
loc:@actor/FullyConnected/b
i
save/RestoreV2_25/tensor_namesConst*
dtype0*3
value*B(Bactor/FullyConnected/b/RMSProp
O
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2
a
save/Assign_25Identitysave/RestoreV2_25*
T0*)
_class
loc:@actor/FullyConnected/b
k
save/RestoreV2_26/tensor_namesConst*
dtype0*5
value,B*B actor/FullyConnected/b/RMSProp_1
O
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2
a
save/Assign_26Identitysave/RestoreV2_26*
T0*)
_class
loc:@actor/FullyConnected/b
c
save/RestoreV2_27/tensor_namesConst*-
value$B"Bactor/FullyConnected_1/W*
dtype0
O
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2
c
save/Assign_27Identitysave/RestoreV2_27*
T0*+
_class!
loc:@actor/FullyConnected_1/W
k
save/RestoreV2_28/tensor_namesConst*5
value,B*B actor/FullyConnected_1/W/RMSProp*
dtype0
O
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2
c
save/Assign_28Identitysave/RestoreV2_28*
T0*+
_class!
loc:@actor/FullyConnected_1/W
m
save/RestoreV2_29/tensor_namesConst*7
value.B,B"actor/FullyConnected_1/W/RMSProp_1*
dtype0
O
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2
c
save/Assign_29Identitysave/RestoreV2_29*
T0*+
_class!
loc:@actor/FullyConnected_1/W
c
save/RestoreV2_30/tensor_namesConst*
dtype0*-
value$B"Bactor/FullyConnected_1/b
O
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2
c
save/Assign_30Identitysave/RestoreV2_30*
T0*+
_class!
loc:@actor/FullyConnected_1/b
k
save/RestoreV2_31/tensor_namesConst*5
value,B*B actor/FullyConnected_1/b/RMSProp*
dtype0
O
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2
c
save/Assign_31Identitysave/RestoreV2_31*
T0*+
_class!
loc:@actor/FullyConnected_1/b
m
save/RestoreV2_32/tensor_namesConst*7
value.B,B"actor/FullyConnected_1/b/RMSProp_1*
dtype0
O
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2
c
save/Assign_32Identitysave/RestoreV2_32*
T0*+
_class!
loc:@actor/FullyConnected_1/b
c
save/RestoreV2_33/tensor_namesConst*
dtype0*-
value$B"Bactor/FullyConnected_2/W
O
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2
c
save/Assign_33Identitysave/RestoreV2_33*
T0*+
_class!
loc:@actor/FullyConnected_2/W
k
save/RestoreV2_34/tensor_namesConst*5
value,B*B actor/FullyConnected_2/W/RMSProp*
dtype0
O
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2
c
save/Assign_34Identitysave/RestoreV2_34*
T0*+
_class!
loc:@actor/FullyConnected_2/W
m
save/RestoreV2_35/tensor_namesConst*
dtype0*7
value.B,B"actor/FullyConnected_2/W/RMSProp_1
O
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2
c
save/Assign_35Identitysave/RestoreV2_35*
T0*+
_class!
loc:@actor/FullyConnected_2/W
c
save/RestoreV2_36/tensor_namesConst*
dtype0*-
value$B"Bactor/FullyConnected_2/b
O
"save/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2
c
save/Assign_36Identitysave/RestoreV2_36*
T0*+
_class!
loc:@actor/FullyConnected_2/b
k
save/RestoreV2_37/tensor_namesConst*5
value,B*B actor/FullyConnected_2/b/RMSProp*
dtype0
O
"save/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2
c
save/Assign_37Identitysave/RestoreV2_37*
T0*+
_class!
loc:@actor/FullyConnected_2/b
m
save/RestoreV2_38/tensor_namesConst*7
value.B,B"actor/FullyConnected_2/b/RMSProp_1*
dtype0
O
"save/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2
c
save/Assign_38Identitysave/RestoreV2_38*
T0*+
_class!
loc:@actor/FullyConnected_2/b
c
save/RestoreV2_39/tensor_namesConst*-
value$B"Bactor/FullyConnected_3/W*
dtype0
O
"save/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2
c
save/Assign_39Identitysave/RestoreV2_39*
T0*+
_class!
loc:@actor/FullyConnected_3/W
k
save/RestoreV2_40/tensor_namesConst*5
value,B*B actor/FullyConnected_3/W/RMSProp*
dtype0
O
"save/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2
c
save/Assign_40Identitysave/RestoreV2_40*
T0*+
_class!
loc:@actor/FullyConnected_3/W
m
save/RestoreV2_41/tensor_namesConst*7
value.B,B"actor/FullyConnected_3/W/RMSProp_1*
dtype0
O
"save/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2
c
save/Assign_41Identitysave/RestoreV2_41*
T0*+
_class!
loc:@actor/FullyConnected_3/W
c
save/RestoreV2_42/tensor_namesConst*-
value$B"Bactor/FullyConnected_3/b*
dtype0
O
"save/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2
c
save/Assign_42Identitysave/RestoreV2_42*
T0*+
_class!
loc:@actor/FullyConnected_3/b
k
save/RestoreV2_43/tensor_namesConst*5
value,B*B actor/FullyConnected_3/b/RMSProp*
dtype0
O
"save/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2
c
save/Assign_43Identitysave/RestoreV2_43*
T0*+
_class!
loc:@actor/FullyConnected_3/b
m
save/RestoreV2_44/tensor_namesConst*7
value.B,B"actor/FullyConnected_3/b/RMSProp_1*
dtype0
O
"save/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2
c
save/Assign_44Identitysave/RestoreV2_44*
T0*+
_class!
loc:@actor/FullyConnected_3/b
c
save/RestoreV2_45/tensor_namesConst*
dtype0*-
value$B"Bactor/FullyConnected_4/W
O
"save/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2
c
save/Assign_45Identitysave/RestoreV2_45*
T0*+
_class!
loc:@actor/FullyConnected_4/W
k
save/RestoreV2_46/tensor_namesConst*
dtype0*5
value,B*B actor/FullyConnected_4/W/RMSProp
O
"save/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2
c
save/Assign_46Identitysave/RestoreV2_46*
T0*+
_class!
loc:@actor/FullyConnected_4/W
m
save/RestoreV2_47/tensor_namesConst*
dtype0*7
value.B,B"actor/FullyConnected_4/W/RMSProp_1
O
"save/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2
c
save/Assign_47Identitysave/RestoreV2_47*
T0*+
_class!
loc:@actor/FullyConnected_4/W
c
save/RestoreV2_48/tensor_namesConst*-
value$B"Bactor/FullyConnected_4/b*
dtype0
O
"save/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2
c
save/Assign_48Identitysave/RestoreV2_48*
T0*+
_class!
loc:@actor/FullyConnected_4/b
k
save/RestoreV2_49/tensor_namesConst*5
value,B*B actor/FullyConnected_4/b/RMSProp*
dtype0
O
"save/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2
c
save/Assign_49Identitysave/RestoreV2_49*
T0*+
_class!
loc:@actor/FullyConnected_4/b
m
save/RestoreV2_50/tensor_namesConst*
dtype0*7
value.B,B"actor/FullyConnected_4/b/RMSProp_1
O
"save/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2
c
save/Assign_50Identitysave/RestoreV2_50*
T0*+
_class!
loc:@actor/FullyConnected_4/b
Z
save/RestoreV2_51/tensor_namesConst*$
valueBBcritic/Conv1D/W*
dtype0
O
"save/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2
Z
save/Assign_51Identitysave/RestoreV2_51*
T0*"
_class
loc:@critic/Conv1D/W
b
save/RestoreV2_52/tensor_namesConst*
dtype0*,
value#B!Bcritic/Conv1D/W/RMSProp
O
"save/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2
Z
save/Assign_52Identitysave/RestoreV2_52*
T0*"
_class
loc:@critic/Conv1D/W
d
save/RestoreV2_53/tensor_namesConst*
dtype0*.
value%B#Bcritic/Conv1D/W/RMSProp_1
O
"save/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2
Z
save/Assign_53Identitysave/RestoreV2_53*
T0*"
_class
loc:@critic/Conv1D/W
Z
save/RestoreV2_54/tensor_namesConst*$
valueBBcritic/Conv1D/b*
dtype0
O
"save/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2
Z
save/Assign_54Identitysave/RestoreV2_54*
T0*"
_class
loc:@critic/Conv1D/b
b
save/RestoreV2_55/tensor_namesConst*,
value#B!Bcritic/Conv1D/b/RMSProp*
dtype0
O
"save/RestoreV2_55/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2
Z
save/Assign_55Identitysave/RestoreV2_55*
T0*"
_class
loc:@critic/Conv1D/b
d
save/RestoreV2_56/tensor_namesConst*.
value%B#Bcritic/Conv1D/b/RMSProp_1*
dtype0
O
"save/RestoreV2_56/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2
Z
save/Assign_56Identitysave/RestoreV2_56*
T0*"
_class
loc:@critic/Conv1D/b
\
save/RestoreV2_57/tensor_namesConst*&
valueBBcritic/Conv1D_1/W*
dtype0
O
"save/RestoreV2_57/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2
\
save/Assign_57Identitysave/RestoreV2_57*
T0*$
_class
loc:@critic/Conv1D_1/W
d
save/RestoreV2_58/tensor_namesConst*.
value%B#Bcritic/Conv1D_1/W/RMSProp*
dtype0
O
"save/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2
\
save/Assign_58Identitysave/RestoreV2_58*
T0*$
_class
loc:@critic/Conv1D_1/W
f
save/RestoreV2_59/tensor_namesConst*0
value'B%Bcritic/Conv1D_1/W/RMSProp_1*
dtype0
O
"save/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2
\
save/Assign_59Identitysave/RestoreV2_59*
T0*$
_class
loc:@critic/Conv1D_1/W
\
save/RestoreV2_60/tensor_namesConst*&
valueBBcritic/Conv1D_1/b*
dtype0
O
"save/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2
\
save/Assign_60Identitysave/RestoreV2_60*
T0*$
_class
loc:@critic/Conv1D_1/b
d
save/RestoreV2_61/tensor_namesConst*.
value%B#Bcritic/Conv1D_1/b/RMSProp*
dtype0
O
"save/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2
\
save/Assign_61Identitysave/RestoreV2_61*
T0*$
_class
loc:@critic/Conv1D_1/b
f
save/RestoreV2_62/tensor_namesConst*0
value'B%Bcritic/Conv1D_1/b/RMSProp_1*
dtype0
O
"save/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_62	RestoreV2
save/Constsave/RestoreV2_62/tensor_names"save/RestoreV2_62/shape_and_slices*
dtypes
2
\
save/Assign_62Identitysave/RestoreV2_62*
T0*$
_class
loc:@critic/Conv1D_1/b
\
save/RestoreV2_63/tensor_namesConst*&
valueBBcritic/Conv1D_2/W*
dtype0
O
"save/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_63	RestoreV2
save/Constsave/RestoreV2_63/tensor_names"save/RestoreV2_63/shape_and_slices*
dtypes
2
\
save/Assign_63Identitysave/RestoreV2_63*
T0*$
_class
loc:@critic/Conv1D_2/W
d
save/RestoreV2_64/tensor_namesConst*.
value%B#Bcritic/Conv1D_2/W/RMSProp*
dtype0
O
"save/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_64	RestoreV2
save/Constsave/RestoreV2_64/tensor_names"save/RestoreV2_64/shape_and_slices*
dtypes
2
\
save/Assign_64Identitysave/RestoreV2_64*
T0*$
_class
loc:@critic/Conv1D_2/W
f
save/RestoreV2_65/tensor_namesConst*0
value'B%Bcritic/Conv1D_2/W/RMSProp_1*
dtype0
O
"save/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_65	RestoreV2
save/Constsave/RestoreV2_65/tensor_names"save/RestoreV2_65/shape_and_slices*
dtypes
2
\
save/Assign_65Identitysave/RestoreV2_65*
T0*$
_class
loc:@critic/Conv1D_2/W
\
save/RestoreV2_66/tensor_namesConst*
dtype0*&
valueBBcritic/Conv1D_2/b
O
"save/RestoreV2_66/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_66	RestoreV2
save/Constsave/RestoreV2_66/tensor_names"save/RestoreV2_66/shape_and_slices*
dtypes
2
\
save/Assign_66Identitysave/RestoreV2_66*
T0*$
_class
loc:@critic/Conv1D_2/b
d
save/RestoreV2_67/tensor_namesConst*.
value%B#Bcritic/Conv1D_2/b/RMSProp*
dtype0
O
"save/RestoreV2_67/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_67	RestoreV2
save/Constsave/RestoreV2_67/tensor_names"save/RestoreV2_67/shape_and_slices*
dtypes
2
\
save/Assign_67Identitysave/RestoreV2_67*
T0*$
_class
loc:@critic/Conv1D_2/b
f
save/RestoreV2_68/tensor_namesConst*0
value'B%Bcritic/Conv1D_2/b/RMSProp_1*
dtype0
O
"save/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_68	RestoreV2
save/Constsave/RestoreV2_68/tensor_names"save/RestoreV2_68/shape_and_slices*
dtypes
2
\
save/Assign_68Identitysave/RestoreV2_68*
T0*$
_class
loc:@critic/Conv1D_2/b
b
save/RestoreV2_69/tensor_namesConst*,
value#B!Bcritic/FullyConnected/W*
dtype0
O
"save/RestoreV2_69/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_69	RestoreV2
save/Constsave/RestoreV2_69/tensor_names"save/RestoreV2_69/shape_and_slices*
dtypes
2
b
save/Assign_69Identitysave/RestoreV2_69*
T0**
_class 
loc:@critic/FullyConnected/W
j
save/RestoreV2_70/tensor_namesConst*4
value+B)Bcritic/FullyConnected/W/RMSProp*
dtype0
O
"save/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_70	RestoreV2
save/Constsave/RestoreV2_70/tensor_names"save/RestoreV2_70/shape_and_slices*
dtypes
2
b
save/Assign_70Identitysave/RestoreV2_70*
T0**
_class 
loc:@critic/FullyConnected/W
l
save/RestoreV2_71/tensor_namesConst*6
value-B+B!critic/FullyConnected/W/RMSProp_1*
dtype0
O
"save/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_71	RestoreV2
save/Constsave/RestoreV2_71/tensor_names"save/RestoreV2_71/shape_and_slices*
dtypes
2
b
save/Assign_71Identitysave/RestoreV2_71*
T0**
_class 
loc:@critic/FullyConnected/W
b
save/RestoreV2_72/tensor_namesConst*,
value#B!Bcritic/FullyConnected/b*
dtype0
O
"save/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_72	RestoreV2
save/Constsave/RestoreV2_72/tensor_names"save/RestoreV2_72/shape_and_slices*
dtypes
2
b
save/Assign_72Identitysave/RestoreV2_72*
T0**
_class 
loc:@critic/FullyConnected/b
j
save/RestoreV2_73/tensor_namesConst*4
value+B)Bcritic/FullyConnected/b/RMSProp*
dtype0
O
"save/RestoreV2_73/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_73	RestoreV2
save/Constsave/RestoreV2_73/tensor_names"save/RestoreV2_73/shape_and_slices*
dtypes
2
b
save/Assign_73Identitysave/RestoreV2_73*
T0**
_class 
loc:@critic/FullyConnected/b
l
save/RestoreV2_74/tensor_namesConst*
dtype0*6
value-B+B!critic/FullyConnected/b/RMSProp_1
O
"save/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_74	RestoreV2
save/Constsave/RestoreV2_74/tensor_names"save/RestoreV2_74/shape_and_slices*
dtypes
2
b
save/Assign_74Identitysave/RestoreV2_74*
T0**
_class 
loc:@critic/FullyConnected/b
d
save/RestoreV2_75/tensor_namesConst*.
value%B#Bcritic/FullyConnected_1/W*
dtype0
O
"save/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_75	RestoreV2
save/Constsave/RestoreV2_75/tensor_names"save/RestoreV2_75/shape_and_slices*
dtypes
2
d
save/Assign_75Identitysave/RestoreV2_75*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
l
save/RestoreV2_76/tensor_namesConst*
dtype0*6
value-B+B!critic/FullyConnected_1/W/RMSProp
O
"save/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_76	RestoreV2
save/Constsave/RestoreV2_76/tensor_names"save/RestoreV2_76/shape_and_slices*
dtypes
2
d
save/Assign_76Identitysave/RestoreV2_76*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
n
save/RestoreV2_77/tensor_namesConst*8
value/B-B#critic/FullyConnected_1/W/RMSProp_1*
dtype0
O
"save/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_77	RestoreV2
save/Constsave/RestoreV2_77/tensor_names"save/RestoreV2_77/shape_and_slices*
dtypes
2
d
save/Assign_77Identitysave/RestoreV2_77*
T0*,
_class"
 loc:@critic/FullyConnected_1/W
d
save/RestoreV2_78/tensor_namesConst*
dtype0*.
value%B#Bcritic/FullyConnected_1/b
O
"save/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_78	RestoreV2
save/Constsave/RestoreV2_78/tensor_names"save/RestoreV2_78/shape_and_slices*
dtypes
2
d
save/Assign_78Identitysave/RestoreV2_78*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
l
save/RestoreV2_79/tensor_namesConst*6
value-B+B!critic/FullyConnected_1/b/RMSProp*
dtype0
O
"save/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_79	RestoreV2
save/Constsave/RestoreV2_79/tensor_names"save/RestoreV2_79/shape_and_slices*
dtypes
2
d
save/Assign_79Identitysave/RestoreV2_79*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
n
save/RestoreV2_80/tensor_namesConst*8
value/B-B#critic/FullyConnected_1/b/RMSProp_1*
dtype0
O
"save/RestoreV2_80/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_80	RestoreV2
save/Constsave/RestoreV2_80/tensor_names"save/RestoreV2_80/shape_and_slices*
dtypes
2
d
save/Assign_80Identitysave/RestoreV2_80*
T0*,
_class"
 loc:@critic/FullyConnected_1/b
d
save/RestoreV2_81/tensor_namesConst*.
value%B#Bcritic/FullyConnected_2/W*
dtype0
O
"save/RestoreV2_81/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_81	RestoreV2
save/Constsave/RestoreV2_81/tensor_names"save/RestoreV2_81/shape_and_slices*
dtypes
2
d
save/Assign_81Identitysave/RestoreV2_81*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
l
save/RestoreV2_82/tensor_namesConst*6
value-B+B!critic/FullyConnected_2/W/RMSProp*
dtype0
O
"save/RestoreV2_82/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_82	RestoreV2
save/Constsave/RestoreV2_82/tensor_names"save/RestoreV2_82/shape_and_slices*
dtypes
2
d
save/Assign_82Identitysave/RestoreV2_82*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
n
save/RestoreV2_83/tensor_namesConst*8
value/B-B#critic/FullyConnected_2/W/RMSProp_1*
dtype0
O
"save/RestoreV2_83/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_83	RestoreV2
save/Constsave/RestoreV2_83/tensor_names"save/RestoreV2_83/shape_and_slices*
dtypes
2
d
save/Assign_83Identitysave/RestoreV2_83*
T0*,
_class"
 loc:@critic/FullyConnected_2/W
d
save/RestoreV2_84/tensor_namesConst*
dtype0*.
value%B#Bcritic/FullyConnected_2/b
O
"save/RestoreV2_84/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_84	RestoreV2
save/Constsave/RestoreV2_84/tensor_names"save/RestoreV2_84/shape_and_slices*
dtypes
2
d
save/Assign_84Identitysave/RestoreV2_84*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
l
save/RestoreV2_85/tensor_namesConst*6
value-B+B!critic/FullyConnected_2/b/RMSProp*
dtype0
O
"save/RestoreV2_85/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_85	RestoreV2
save/Constsave/RestoreV2_85/tensor_names"save/RestoreV2_85/shape_and_slices*
dtypes
2
d
save/Assign_85Identitysave/RestoreV2_85*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
n
save/RestoreV2_86/tensor_namesConst*8
value/B-B#critic/FullyConnected_2/b/RMSProp_1*
dtype0
O
"save/RestoreV2_86/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_86	RestoreV2
save/Constsave/RestoreV2_86/tensor_names"save/RestoreV2_86/shape_and_slices*
dtypes
2
d
save/Assign_86Identitysave/RestoreV2_86*
T0*,
_class"
 loc:@critic/FullyConnected_2/b
d
save/RestoreV2_87/tensor_namesConst*.
value%B#Bcritic/FullyConnected_3/W*
dtype0
O
"save/RestoreV2_87/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_87	RestoreV2
save/Constsave/RestoreV2_87/tensor_names"save/RestoreV2_87/shape_and_slices*
dtypes
2
d
save/Assign_87Identitysave/RestoreV2_87*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
l
save/RestoreV2_88/tensor_namesConst*6
value-B+B!critic/FullyConnected_3/W/RMSProp*
dtype0
O
"save/RestoreV2_88/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_88	RestoreV2
save/Constsave/RestoreV2_88/tensor_names"save/RestoreV2_88/shape_and_slices*
dtypes
2
d
save/Assign_88Identitysave/RestoreV2_88*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
n
save/RestoreV2_89/tensor_namesConst*8
value/B-B#critic/FullyConnected_3/W/RMSProp_1*
dtype0
O
"save/RestoreV2_89/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_89	RestoreV2
save/Constsave/RestoreV2_89/tensor_names"save/RestoreV2_89/shape_and_slices*
dtypes
2
d
save/Assign_89Identitysave/RestoreV2_89*
T0*,
_class"
 loc:@critic/FullyConnected_3/W
d
save/RestoreV2_90/tensor_namesConst*.
value%B#Bcritic/FullyConnected_3/b*
dtype0
O
"save/RestoreV2_90/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_90	RestoreV2
save/Constsave/RestoreV2_90/tensor_names"save/RestoreV2_90/shape_and_slices*
dtypes
2
d
save/Assign_90Identitysave/RestoreV2_90*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
l
save/RestoreV2_91/tensor_namesConst*
dtype0*6
value-B+B!critic/FullyConnected_3/b/RMSProp
O
"save/RestoreV2_91/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_91	RestoreV2
save/Constsave/RestoreV2_91/tensor_names"save/RestoreV2_91/shape_and_slices*
dtypes
2
d
save/Assign_91Identitysave/RestoreV2_91*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
n
save/RestoreV2_92/tensor_namesConst*
dtype0*8
value/B-B#critic/FullyConnected_3/b/RMSProp_1
O
"save/RestoreV2_92/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_92	RestoreV2
save/Constsave/RestoreV2_92/tensor_names"save/RestoreV2_92/shape_and_slices*
dtypes
2
d
save/Assign_92Identitysave/RestoreV2_92*
T0*,
_class"
 loc:@critic/FullyConnected_3/b
d
save/RestoreV2_93/tensor_namesConst*.
value%B#Bcritic/FullyConnected_4/W*
dtype0
O
"save/RestoreV2_93/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_93	RestoreV2
save/Constsave/RestoreV2_93/tensor_names"save/RestoreV2_93/shape_and_slices*
dtypes
2
d
save/Assign_93Identitysave/RestoreV2_93*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
l
save/RestoreV2_94/tensor_namesConst*6
value-B+B!critic/FullyConnected_4/W/RMSProp*
dtype0
O
"save/RestoreV2_94/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_94	RestoreV2
save/Constsave/RestoreV2_94/tensor_names"save/RestoreV2_94/shape_and_slices*
dtypes
2
d
save/Assign_94Identitysave/RestoreV2_94*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
n
save/RestoreV2_95/tensor_namesConst*
dtype0*8
value/B-B#critic/FullyConnected_4/W/RMSProp_1
O
"save/RestoreV2_95/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_95	RestoreV2
save/Constsave/RestoreV2_95/tensor_names"save/RestoreV2_95/shape_and_slices*
dtypes
2
d
save/Assign_95Identitysave/RestoreV2_95*
T0*,
_class"
 loc:@critic/FullyConnected_4/W
d
save/RestoreV2_96/tensor_namesConst*.
value%B#Bcritic/FullyConnected_4/b*
dtype0
O
"save/RestoreV2_96/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_96	RestoreV2
save/Constsave/RestoreV2_96/tensor_names"save/RestoreV2_96/shape_and_slices*
dtypes
2
d
save/Assign_96Identitysave/RestoreV2_96*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
l
save/RestoreV2_97/tensor_namesConst*
dtype0*6
value-B+B!critic/FullyConnected_4/b/RMSProp
O
"save/RestoreV2_97/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_97	RestoreV2
save/Constsave/RestoreV2_97/tensor_names"save/RestoreV2_97/shape_and_slices*
dtypes
2
d
save/Assign_97Identitysave/RestoreV2_97*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
n
save/RestoreV2_98/tensor_namesConst*8
value/B-B#critic/FullyConnected_4/b/RMSProp_1*
dtype0
O
"save/RestoreV2_98/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_98	RestoreV2
save/Constsave/RestoreV2_98/tensor_names"save/RestoreV2_98/shape_and_slices*
dtypes
2
d
save/Assign_98Identitysave/RestoreV2_98*
T0*,
_class"
 loc:@critic/FullyConnected_4/b
V
save/RestoreV2_99/tensor_namesConst* 
valueBBis_training*
dtype0
O
"save/RestoreV2_99/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_99	RestoreV2
save/Constsave/RestoreV2_99/tensor_names"save/RestoreV2_99/shape_and_slices*
dtypes
2

V
save/Assign_99Identitysave/RestoreV2_99*
T0
*
_class
loc:@is_training
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99 