ПТ(
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02unknown8╖▌&
К
embedding_5/embeddingsVarHandleOp*
shape:
Ущd*'
shared_nameembedding_5/embeddings*
dtype0*
_output_shapes
: 
Г
*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings*
dtype0* 
_output_shapes
:
Ущd
x
dense_8/kernelVarHandleOp*
shape
:d*
shared_namedense_8/kernel*
dtype0*
_output_shapes
: 
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
dtype0*
_output_shapes

:d
p
dense_8/biasVarHandleOp*
shape:*
shared_namedense_8/bias*
dtype0*
_output_shapes
: 
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
u
gru_5/kernelVarHandleOp*
shape:	dм*
shared_namegru_5/kernel*
dtype0*
_output_shapes
: 
n
 gru_5/kernel/Read/ReadVariableOpReadVariableOpgru_5/kernel*
dtype0*
_output_shapes
:	dм
Й
gru_5/recurrent_kernelVarHandleOp*
shape:	dм*'
shared_namegru_5/recurrent_kernel*
dtype0*
_output_shapes
: 
В
*gru_5/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_5/recurrent_kernel*
dtype0*
_output_shapes
:	dм
q

gru_5/biasVarHandleOp*
shape:	м*
shared_name
gru_5/bias*
dtype0*
_output_shapes
: 
j
gru_5/bias/Read/ReadVariableOpReadVariableOp
gru_5/bias*
dtype0*
_output_shapes
:	м
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
Ш
Adam/embedding_5/embeddings/mVarHandleOp*
shape:
Ущd*.
shared_nameAdam/embedding_5/embeddings/m*
dtype0*
_output_shapes
: 
С
1Adam/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_5/embeddings/m*
dtype0* 
_output_shapes
:
Ущd
Ж
Adam/dense_8/kernel/mVarHandleOp*
shape
:d*&
shared_nameAdam/dense_8/kernel/m*
dtype0*
_output_shapes
: 

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
dtype0*
_output_shapes

:d
~
Adam/dense_8/bias/mVarHandleOp*
shape:*$
shared_nameAdam/dense_8/bias/m*
dtype0*
_output_shapes
: 
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
dtype0*
_output_shapes
:
Г
Adam/gru_5/kernel/mVarHandleOp*
shape:	dм*$
shared_nameAdam/gru_5/kernel/m*
dtype0*
_output_shapes
: 
|
'Adam/gru_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_5/kernel/m*
dtype0*
_output_shapes
:	dм
Ч
Adam/gru_5/recurrent_kernel/mVarHandleOp*
shape:	dм*.
shared_nameAdam/gru_5/recurrent_kernel/m*
dtype0*
_output_shapes
: 
Р
1Adam/gru_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_5/recurrent_kernel/m*
dtype0*
_output_shapes
:	dм

Adam/gru_5/bias/mVarHandleOp*
shape:	м*"
shared_nameAdam/gru_5/bias/m*
dtype0*
_output_shapes
: 
x
%Adam/gru_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_5/bias/m*
dtype0*
_output_shapes
:	м
Ш
Adam/embedding_5/embeddings/vVarHandleOp*
shape:
Ущd*.
shared_nameAdam/embedding_5/embeddings/v*
dtype0*
_output_shapes
: 
С
1Adam/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_5/embeddings/v*
dtype0* 
_output_shapes
:
Ущd
Ж
Adam/dense_8/kernel/vVarHandleOp*
shape
:d*&
shared_nameAdam/dense_8/kernel/v*
dtype0*
_output_shapes
: 

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
dtype0*
_output_shapes

:d
~
Adam/dense_8/bias/vVarHandleOp*
shape:*$
shared_nameAdam/dense_8/bias/v*
dtype0*
_output_shapes
: 
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
dtype0*
_output_shapes
:
Г
Adam/gru_5/kernel/vVarHandleOp*
shape:	dм*$
shared_nameAdam/gru_5/kernel/v*
dtype0*
_output_shapes
: 
|
'Adam/gru_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_5/kernel/v*
dtype0*
_output_shapes
:	dм
Ч
Adam/gru_5/recurrent_kernel/vVarHandleOp*
shape:	dм*.
shared_nameAdam/gru_5/recurrent_kernel/v*
dtype0*
_output_shapes
: 
Р
1Adam/gru_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_5/recurrent_kernel/v*
dtype0*
_output_shapes
:	dм

Adam/gru_5/bias/vVarHandleOp*
shape:	м*"
shared_nameAdam/gru_5/bias/v*
dtype0*
_output_shapes
: 
x
%Adam/gru_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_5/bias/v*
dtype0*
_output_shapes
:	м

NoOpNoOp
Д)
ConstConst"/device:CPU:0*┐(
value╡(B▓( Bл(
є
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
м
 iter

!beta_1

"beta_2
	#decay
$learning_ratemPmQmR%mS&mT'mUvVvWvX%vY&vZ'v[
*
0
%1
&2
'3
4
5
 
*
0
%1
&2
'3
4
5
Ъ
(non_trainable_variables
trainable_variables
regularization_losses
	variables

)layers
*layer_regularization_losses
+metrics
 
 
 
 
Ъ
,non_trainable_variables
trainable_variables
regularization_losses
	variables

-layers
.layer_regularization_losses
/metrics
fd
VARIABLE_VALUEembedding_5/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
Ъ
0non_trainable_variables
trainable_variables
regularization_losses
	variables

1layers
2layer_regularization_losses
3metrics
~

%kernel
&recurrent_kernel
'bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
 

%0
&1
'2
 

%0
&1
'2
Ъ
8non_trainable_variables
trainable_variables
regularization_losses
	variables

9layers
:layer_regularization_losses
;metrics
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Ъ
<non_trainable_variables
trainable_variables
regularization_losses
	variables

=layers
>layer_regularization_losses
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEgru_5/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEgru_5/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
gru_5/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 

@0
 
 
 
 
 
 
 
 

%0
&1
'2
 

%0
&1
'2
Ъ
Anon_trainable_variables
4trainable_variables
5regularization_losses
6	variables

Blayers
Clayer_regularization_losses
Dmetrics
 

0
 
 
 
 
 
 
x
	Etotal
	Fcount
G
_fn_kwargs
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

E0
F1
Ъ
Lnon_trainable_variables
Htrainable_variables
Iregularization_losses
J	variables

Mlayers
Nlayer_regularization_losses
Ometrics

E0
F1
 
 
 
КЗ
VARIABLE_VALUEAdam/embedding_5/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/gru_5/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/gru_5/recurrent_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/gru_5/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUEAdam/embedding_5/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/gru_5/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/gru_5/recurrent_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/gru_5/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Ж
!serving_default_embedding_5_inputPlaceholder*
shape:         щ*
dtype0*(
_output_shapes
:         щ
Ц
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_5_inputembedding_5/embeddingsgru_5/kernelgru_5/recurrent_kernel
gru_5/biasdense_8/kerneldense_8/bias*-
_gradient_op_typePartitionedCall-211271*-
f(R&
$__inference_signature_wrapper_208636*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
ї	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_5/embeddings/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp gru_5/kernel/Read/ReadVariableOp*gru_5/recurrent_kernel/Read/ReadVariableOpgru_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_5/embeddings/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp'Adam/gru_5/kernel/m/Read/ReadVariableOp1Adam/gru_5/recurrent_kernel/m/Read/ReadVariableOp%Adam/gru_5/bias/m/Read/ReadVariableOp1Adam/embedding_5/embeddings/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp'Adam/gru_5/kernel/v/Read/ReadVariableOp1Adam/gru_5/recurrent_kernel/v/Read/ReadVariableOp%Adam/gru_5/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-211318*(
f#R!
__inference__traced_save_211317*
Tout
2*-
config_proto

CPU

GPU2*0J 8*&
Tin
2	*
_output_shapes
: 
№
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_5/embeddingsdense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_5/kernelgru_5/recurrent_kernel
gru_5/biastotalcountAdam/embedding_5/embeddings/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/gru_5/kernel/mAdam/gru_5/recurrent_kernel/mAdam/gru_5/bias/mAdam/embedding_5/embeddings/vAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/gru_5/kernel/vAdam/gru_5/recurrent_kernel/vAdam/gru_5/bias/v*-
_gradient_op_typePartitionedCall-211406*+
f&R$
"__inference__traced_restore_211405*
Tout
2*-
config_proto

CPU

GPU2*0J 8*%
Tin
2*
_output_shapes
: зя%
°
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_208084

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dм
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-207854*(
f#R!
__inference_standard_gru_207853*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
щ>
╢
(__forward_cudnn_gru_with_fallback_209478

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_209340_209479*<
api_implements*(gru_3c4c0838-5707-4c61-9ea1-6433c1f0c280*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╖b
и
"__inference__traced_restore_211405
file_prefix+
'assignvariableop_embedding_5_embeddings%
!assignvariableop_1_dense_8_kernel#
assignvariableop_2_dense_8_bias 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate#
assignvariableop_8_gru_5_kernel-
)assignvariableop_9_gru_5_recurrent_kernel"
assignvariableop_10_gru_5_bias
assignvariableop_11_total
assignvariableop_12_count5
1assignvariableop_13_adam_embedding_5_embeddings_m-
)assignvariableop_14_adam_dense_8_kernel_m+
'assignvariableop_15_adam_dense_8_bias_m+
'assignvariableop_16_adam_gru_5_kernel_m5
1assignvariableop_17_adam_gru_5_recurrent_kernel_m)
%assignvariableop_18_adam_gru_5_bias_m5
1assignvariableop_19_adam_embedding_5_embeddings_v-
)assignvariableop_20_adam_dense_8_kernel_v+
'assignvariableop_21_adam_dense_8_bias_v+
'assignvariableop_22_adam_gru_5_kernel_v5
1assignvariableop_23_adam_gru_5_recurrent_kernel_v)
%assignvariableop_24_adam_gru_5_bias_v
identity_26ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1╨
RestoreV2/tensor_namesConst"/device:CPU:0*Ў
valueьBщB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:в
RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*'
dtypes
2	*x
_output_shapesf
d:::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp'assignvariableop_embedding_5_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:Б
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_8_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_8_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0	*
_output_shapes
:|
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0*
dtype0	*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:~
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:~
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:}
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Е
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_gru_5_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:Й
AssignVariableOp_9AssignVariableOp)assignvariableop_9_gru_5_recurrent_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:А
AssignVariableOp_10AssignVariableOpassignvariableop_10_gru_5_biasIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:{
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:У
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_embedding_5_embeddings_mIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Л
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_8_kernel_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Й
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_8_bias_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Й
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_gru_5_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:У
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_gru_5_recurrent_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:З
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_gru_5_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_5_embeddings_vIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Л
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_8_kernel_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:Й
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_8_bias_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:Й
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_gru_5_kernel_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:У
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_gru_5_recurrent_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:З
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_gru_5_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ї
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: В
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : :
 : : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : 
Ч
╒
H__inference_sequential_5_layer_call_and_return_conditional_losses_208605

inputs.
*embedding_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3*
&dense_8_statefulpartitionedcall_args_1*
&dense_8_statefulpartitionedcall_args_2
identityИвdense_8/StatefulPartitionedCallв#embedding_5/StatefulPartitionedCallвgru_5/StatefulPartitionedCallю
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs*embedding_5_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-207665*P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*,
_output_shapes
:         щd╦
gru_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0$gru_5_statefulpartitionedcall_args_1$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208507*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208495*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dж
dense_8/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0&dense_8_statefulpartitionedcall_args_1&dense_8_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-208535*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_208529*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         ╪
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
И
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_207217

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         d┤
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-206987*(
f#R!
__inference_standard_gru_206986*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*\
_output_shapesJ
H:         d:                  d:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
┴	
╔
-__inference_sequential_5_layer_call_fn_208588
embedding_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-208579*Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_208578*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
▐+
├
while_body_209142
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
°
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_208495

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dм
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-208265*(
f#R!
__inference_standard_gru_208264*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
юУ
┘
<__inference___backward_cudnn_gru_with_fallback_211042_211181
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         dm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  d`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╜
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dВ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:К
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  du
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мж
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_55dd13e9-718a-4cf5-a8ce-218843da5b97*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_211180*╩
_input_shapes╕
╡:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_209797_209936
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0913c5d8-d932-4763-a533-8c91bc1de75b*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_209935*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ў
Л
while_cond_208716
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_208915_209054
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_90fbe536-460f-4ad5-8b42-e9d4c2ce9f54*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_209053*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
Ч
╒
H__inference_sequential_5_layer_call_and_return_conditional_losses_208578

inputs.
*embedding_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3*
&dense_8_statefulpartitionedcall_args_1*
&dense_8_statefulpartitionedcall_args_2
identityИвdense_8/StatefulPartitionedCallв#embedding_5/StatefulPartitionedCallвgru_5/StatefulPartitionedCallю
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs*embedding_5_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-207665*P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*,
_output_shapes
:         щd╦
gru_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0$gru_5_statefulpartitionedcall_args_1$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208498*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208084*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dж
dense_8/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0&dense_8_statefulpartitionedcall_args_1&dense_8_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-208535*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_208529*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         ╪
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
юУ
┘
<__inference___backward_cudnn_gru_with_fallback_210633_210772
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         dm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  d`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╜
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dВ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:К
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  du
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мж
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_2a021932-19b6-4b8b-972b-6d94dfd4e5b3*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_210771*╩
_input_shapes╕
╡:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
╬	
▄
C__inference_dense_8_layer_call_and_return_conditional_losses_208529

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:di
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
д0
Е
*__inference_cudnn_gru_with_fallback_210207

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0f6aeb4c-5059-4758-8433-370b7b79cd9d*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
щ>
╢
(__forward_cudnn_gru_with_fallback_209053

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_208915_209054*<
api_implements*(gru_90fbe536-460f-4ad5-8b42-e9d4c2ce9f54*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
й
╦
&__inference_gru_5_layer_call_fn_210357

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208498*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208084*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dВ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
╓J
ё
__inference_standard_gru_206986

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_206879*
_num_original_outputs*
bodyR
while_body_206880*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  dо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_7563a40a-6cc7-4cfd-8257-ba857808c14f*
api_preferred_deviceCPU*R
_input_shapesA
?:                  d:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
щ>
╢
(__forward_cudnn_gru_with_fallback_205972

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_205834_205973*<
api_implements*(gru_62d9dea7-8d5e-4d01-b597-d07fc355c364*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
жJ
ё
__inference_standard_gru_205744

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_205635*
_num_original_outputs*
bodyR
while_body_205636*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62d9dea7-8d5e-4d01-b597-d07fc355c364*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
Р
ш
A__inference_gru_5_layer_call_and_return_conditional_losses_211183
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         d╢
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-210953*(
f#R!
__inference_standard_gru_210952*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*\
_output_shapesJ
H:         d:                  d:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
▐+
├
while_body_210010
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
жJ
ё
__inference_standard_gru_209707

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_209598*
_num_original_outputs*
bodyR
while_body_209599*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0913c5d8-d932-4763-a533-8c91bc1de75b*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
И
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_207637

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         d┤
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-207407*(
f#R!
__inference_standard_gru_207406*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*\
_output_shapesJ
H:         d:                  d:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ў
Л
while_cond_207744
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
жJ
ё
__inference_standard_gru_208825

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_208716*
_num_original_outputs*
bodyR
while_body_208717*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_90fbe536-460f-4ad5-8b42-e9d4c2ce9f54*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
а	
╛
-__inference_sequential_5_layer_call_fn_209499

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-208579*Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_208578*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
ў
Л
while_cond_205635
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
Э
Й
,__inference_embedding_5_layer_call_fn_209527

inputs"
statefulpartitionedcall_args_1
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-207665*P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*,
_output_shapes
:         щdЗ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         щd"
identityIdentity:output:0*+
_input_shapes
:         щ:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
жJ
ё
__inference_standard_gru_210118

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_210009*
_num_original_outputs*
bodyR
while_body_210010*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0f6aeb4c-5059-4758-8433-370b7b79cd9d*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╖
═
&__inference_gru_5_layer_call_fn_211199
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-207638*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_207637*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dВ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
╓J
ё
__inference_standard_gru_210543

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_210436*
_num_original_outputs*
bodyR
while_body_210437*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  dо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_2a021932-19b6-4b8b-972b-6d94dfd4e5b3*
api_preferred_deviceCPU*R
_input_shapesA
?:                  d:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
▐+
├
while_body_210846
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
д0
Е
*__inference_cudnn_gru_with_fallback_208914

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_90fbe536-460f-4ad5-8b42-e9d4c2ce9f54*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
щ>
╢
(__forward_cudnn_gru_with_fallback_208081

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_207943_208082*<
api_implements*(gru_dd2c674e-9182-4891-8d78-cfc7b4d6e4a7*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╖
═
&__inference_gru_5_layer_call_fn_211191
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-207218*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_207217*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dВ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
щ>
╢
(__forward_cudnn_gru_with_fallback_207634

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_207496_207635*<
api_implements*(gru_327f3b28-e868-4060-9d14-c5b7bfead4f0*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╩
Л
while_cond_210436
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
жJ
ё
__inference_standard_gru_208264

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_208155*
_num_original_outputs*
bodyR
while_body_208156*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_22bdfb2f-1234-4e20-ab6f-690e46f89e31*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
▐+
├
while_body_209599
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
С	
└
$__inference_signature_wrapper_208636
embedding_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-208627**
f%R#
!__inference__wrapped_model_205982*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_207943_208082
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_dd2c674e-9182-4891-8d78-cfc7b4d6e4a7*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_208081*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_210208_210347
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0f6aeb4c-5059-4758-8433-370b7b79cd9d*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_210346*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
▐+
├
while_body_206880
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
░
╙
G__inference_embedding_5_layer_call_and_return_conditional_losses_209521

inputs1
-embedding_lookup_read_readvariableop_resource
identityИвembedding_lookupв$embedding_lookup/Read/ReadVariableOpV
CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         щ┬
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Ущd~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ущd┤
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceCast:y:0%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         щdр
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         щdД
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         щdл
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*,
_output_shapes
:         щd"
identityIdentity:output:0*+
_input_shapes
:         щ:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
▐+
├
while_body_208717
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
юУ
┘
<__inference___backward_cudnn_gru_with_fallback_207076_207215
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         dm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  d`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╜
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dВ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:К
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  du
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мж
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_7563a40a-6cc7-4cfd-8257-ba857808c14f*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_207214*╩
_input_shapes╕
╡:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
╬	
▄
C__inference_dense_8_layer_call_and_return_conditional_losses_211210

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:di
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
д0
Е
*__inference_cudnn_gru_with_fallback_208353

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_22bdfb2f-1234-4e20-ab6f-690e46f89e31*
api_preferred_deviceGPU*
_input_shapes 20
Reshape/ReadVariableOpReshape/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_208354_208493
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_22bdfb2f-1234-4e20-ab6f-690e46f89e31*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_208492*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
░
╙
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659

inputs1
-embedding_lookup_read_readvariableop_resource
identityИвembedding_lookupв$embedding_lookup/Read/ReadVariableOpV
CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         щ┬
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Ущd~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ущd┤
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceCast:y:0%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         щdр
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         щdД
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         щdл
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*,
_output_shapes
:         щd"
identityIdentity:output:0*+
_input_shapes
:         щ:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
°
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_210349

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dм
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-210119*(
f#R!
__inference_standard_gru_210118*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
д0
Е
*__inference_cudnn_gru_with_fallback_210632

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_2a021932-19b6-4b8b-972b-6d94dfd4e5b3*
api_preferred_deviceGPU*
_input_shapes 20
Reshape/ReadVariableOpReshape/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_205834_205973
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62d9dea7-8d5e-4d01-b597-d07fc355c364*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_205972*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
ў
Л
while_cond_208155
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
д0
Е
*__inference_cudnn_gru_with_fallback_209796

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_0913c5d8-d932-4763-a533-8c91bc1de75b*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
щ>
╢
(__forward_cudnn_gru_with_fallback_208492

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_208354_208493*<
api_implements*(gru_22bdfb2f-1234-4e20-ab6f-690e46f89e31*
api_preferred_deviceGPU*
_input_shapes 20
Reshape/ReadVariableOpReshape/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╓J
ё
__inference_standard_gru_207406

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_207299*
_num_original_outputs*
bodyR
while_body_207300*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  dо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_327f3b28-e868-4060-9d14-c5b7bfead4f0*
api_preferred_deviceCPU*R
_input_shapesA
?:                  d:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
щ>
╢
(__forward_cudnn_gru_with_fallback_209935

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_209797_209936*<
api_implements*(gru_0913c5d8-d932-4763-a533-8c91bc1de75b*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╩
Л
while_cond_207299
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
У*
░
H__inference_sequential_5_layer_call_and_return_conditional_losses_209488

inputs=
9embedding_5_embedding_lookup_read_readvariableop_resource(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3(
$gru_5_statefulpartitionedcall_args_4*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвembedding_5/embedding_lookupв0embedding_5/embedding_lookup/Read/ReadVariableOpвgru_5/StatefulPartitionedCallb
embedding_5/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         щ┌
0embedding_5/embedding_lookup/Read/ReadVariableOpReadVariableOp9embedding_5_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
УщdЦ
%embedding_5/embedding_lookup/IdentityIdentity8embedding_5/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
УщdЁ
embedding_5/embedding_lookupResourceGather9embedding_5_embedding_lookup_read_readvariableop_resourceembedding_5/Cast:y:01^embedding_5/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*C
_class9
75loc:@embedding_5/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         щdД
'embedding_5/embedding_lookup/Identity_1Identity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@embedding_5/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         щdЬ
'embedding_5/embedding_lookup/Identity_2Identity0embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         щdk
gru_5/ShapeShape0embedding_5/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:c
gru_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:я
gru_5/strided_sliceStridedSlicegru_5/Shape:output:0"gru_5/strided_slice/stack:output:0$gru_5/strided_slice/stack_1:output:0$gru_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: S
gru_5/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: q
gru_5/zeros/mulMulgru_5/strided_slice:output:0gru_5/zeros/mul/y:output:0*
T0*
_output_shapes
: U
gru_5/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: k
gru_5/zeros/LessLessgru_5/zeros/mul:z:0gru_5/zeros/Less/y:output:0*
T0*
_output_shapes
: V
gru_5/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: Е
gru_5/zeros/packedPackgru_5/strided_slice:output:0gru_5/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:V
gru_5/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ~
gru_5/zerosFillgru_5/zeros/packed:output:0gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:         dЇ
gru_5/StatefulPartitionedCallStatefulPartitionedCall0embedding_5/embedding_lookup/Identity_2:output:0gru_5/zeros:output:0$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3$gru_5_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-209251*(
f#R!
__inference_standard_gru_209250*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: ▓
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:dЩ
dense_8/MatMulMatMul&gru_5/StatefulPartitionedCall:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ░
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         О
IdentityIdentitydense_8/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^embedding_5/embedding_lookup1^embedding_5/embedding_lookup/Read/ReadVariableOp^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2d
0embedding_5/embedding_lookup/Read/ReadVariableOp0embedding_5/embedding_lookup/Read/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2<
embedding_5/embedding_lookupembedding_5/embedding_lookup: : : : :& "
 
_user_specified_nameinputs: : 
щ>
╢
(__forward_cudnn_gru_with_fallback_210346

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_210208_210347*<
api_implements*(gru_0f6aeb4c-5059-4758-8433-370b7b79cd9d*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
ў
Л
while_cond_209141
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
ў
Л
while_cond_209598
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
а	
╛
-__inference_sequential_5_layer_call_fn_209510

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-208606*Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_208605*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
°
ц
A__inference_gru_5_layer_call_and_return_conditional_losses_209938

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         dм
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-209708*(
f#R!
__inference_standard_gru_209707*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
▐+
├
while_body_205636
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
Ё2
г
!__inference__wrapped_model_205982
embedding_5_inputJ
Fsequential_5_embedding_5_embedding_lookup_read_readvariableop_resource5
1sequential_5_gru_5_statefulpartitionedcall_args_25
1sequential_5_gru_5_statefulpartitionedcall_args_35
1sequential_5_gru_5_statefulpartitionedcall_args_47
3sequential_5_dense_8_matmul_readvariableop_resource8
4sequential_5_dense_8_biasadd_readvariableop_resource
identityИв+sequential_5/dense_8/BiasAdd/ReadVariableOpв*sequential_5/dense_8/MatMul/ReadVariableOpв)sequential_5/embedding_5/embedding_lookupв=sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOpв*sequential_5/gru_5/StatefulPartitionedCallz
sequential_5/embedding_5/CastCastembedding_5_input*

SrcT0*

DstT0*(
_output_shapes
:         щЇ
=sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOpReadVariableOpFsequential_5_embedding_5_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Ущd░
2sequential_5/embedding_5/embedding_lookup/IdentityIdentityEsequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ущd▒
)sequential_5/embedding_5/embedding_lookupResourceGatherFsequential_5_embedding_5_embedding_lookup_read_readvariableop_resource!sequential_5/embedding_5/Cast:y:0>^sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         щdл
4sequential_5/embedding_5/embedding_lookup/Identity_1Identity2sequential_5/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         щd╢
4sequential_5/embedding_5/embedding_lookup/Identity_2Identity=sequential_5/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         щdЕ
sequential_5/gru_5/ShapeShape=sequential_5/embedding_5/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:p
&sequential_5/gru_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:r
(sequential_5/gru_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:r
(sequential_5/gru_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:░
 sequential_5/gru_5/strided_sliceStridedSlice!sequential_5/gru_5/Shape:output:0/sequential_5/gru_5/strided_slice/stack:output:01sequential_5/gru_5/strided_slice/stack_1:output:01sequential_5/gru_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: `
sequential_5/gru_5/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: Ш
sequential_5/gru_5/zeros/mulMul)sequential_5/gru_5/strided_slice:output:0'sequential_5/gru_5/zeros/mul/y:output:0*
T0*
_output_shapes
: b
sequential_5/gru_5/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Т
sequential_5/gru_5/zeros/LessLess sequential_5/gru_5/zeros/mul:z:0(sequential_5/gru_5/zeros/Less/y:output:0*
T0*
_output_shapes
: c
!sequential_5/gru_5/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: м
sequential_5/gru_5/zeros/packedPack)sequential_5/gru_5/strided_slice:output:0*sequential_5/gru_5/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:c
sequential_5/gru_5/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: е
sequential_5/gru_5/zerosFill(sequential_5/gru_5/zeros/packed:output:0'sequential_5/gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:         d┬
*sequential_5/gru_5/StatefulPartitionedCallStatefulPartitionedCall=sequential_5/embedding_5/embedding_lookup/Identity_2:output:0!sequential_5/gru_5/zeros:output:01sequential_5_gru_5_statefulpartitionedcall_args_21sequential_5_gru_5_statefulpartitionedcall_args_31sequential_5_gru_5_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-205745*(
f#R!
__inference_standard_gru_205744*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: ╠
*sequential_5/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_8_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:d└
sequential_5/dense_8/MatMulMatMul3sequential_5/gru_5/StatefulPartitionedCall:output:02sequential_5/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╩
+sequential_5/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_8_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:╡
sequential_5/dense_8/BiasAddBiasAdd%sequential_5/dense_8/MatMul:product:03sequential_5/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_5/dense_8/SigmoidSigmoid%sequential_5/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         ▄
IdentityIdentity sequential_5/dense_8/Sigmoid:y:0,^sequential_5/dense_8/BiasAdd/ReadVariableOp+^sequential_5/dense_8/MatMul/ReadVariableOp*^sequential_5/embedding_5/embedding_lookup>^sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp+^sequential_5/gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2~
=sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp=sequential_5/embedding_5/embedding_lookup/Read/ReadVariableOp2Z
+sequential_5/dense_8/BiasAdd/ReadVariableOp+sequential_5/dense_8/BiasAdd/ReadVariableOp2V
)sequential_5/embedding_5/embedding_lookup)sequential_5/embedding_5/embedding_lookup2X
*sequential_5/gru_5/StatefulPartitionedCall*sequential_5/gru_5/StatefulPartitionedCall2X
*sequential_5/dense_8/MatMul/ReadVariableOp*sequential_5/dense_8/MatMul/ReadVariableOp: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
╕
р
H__inference_sequential_5_layer_call_and_return_conditional_losses_208562
embedding_5_input.
*embedding_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3*
&dense_8_statefulpartitionedcall_args_1*
&dense_8_statefulpartitionedcall_args_2
identityИвdense_8/StatefulPartitionedCallв#embedding_5/StatefulPartitionedCallвgru_5/StatefulPartitionedCall∙
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallembedding_5_input*embedding_5_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-207665*P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*,
_output_shapes
:         щd╦
gru_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0$gru_5_statefulpartitionedcall_args_1$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208507*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208495*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dж
dense_8/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0&dense_8_statefulpartitionedcall_args_1&dense_8_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-208535*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_208529*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         ╪
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
▐+
├
while_body_208156
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
щ>
╢
(__forward_cudnn_gru_with_fallback_211180

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_211042_211181*<
api_implements*(gru_55dd13e9-718a-4cf5-a8ce-218843da5b97*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
Р
ш
A__inference_gru_5_layer_call_and_return_conditional_losses_210774
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         d╢
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-210544*(
f#R!
__inference_standard_gru_210543*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*\
_output_shapesJ
H:         d:                  d:         d: В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :( $
"
_user_specified_name
inputs/0: : 
д0
Е
*__inference_cudnn_gru_with_fallback_211041

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_55dd13e9-718a-4cf5-a8ce-218843da5b97*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
И7
ш

__inference__traced_save_211317
file_prefix5
1savev2_embedding_5_embeddings_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop+
'savev2_gru_5_kernel_read_readvariableop5
1savev2_gru_5_recurrent_kernel_read_readvariableop)
%savev2_gru_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop<
8savev2_adam_embedding_5_embeddings_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop2
.savev2_adam_gru_5_kernel_m_read_readvariableop<
8savev2_adam_gru_5_recurrent_kernel_m_read_readvariableop0
,savev2_adam_gru_5_bias_m_read_readvariableop<
8savev2_adam_embedding_5_embeddings_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop2
.savev2_adam_gru_5_kernel_v_read_readvariableop<
8savev2_adam_gru_5_recurrent_kernel_v_read_readvariableop0
,savev2_adam_gru_5_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5597960ca0b845af9ead5aa46d7c2e0d/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ═
SaveV2/tensor_namesConst"/device:CPU:0*Ў
valueьBщB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Я
SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:╝

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_5_embeddings_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop'savev2_gru_5_kernel_read_readvariableop1savev2_gru_5_recurrent_kernel_read_readvariableop%savev2_gru_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_5_embeddings_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop.savev2_adam_gru_5_kernel_m_read_readvariableop8savev2_adam_gru_5_recurrent_kernel_m_read_readvariableop,savev2_adam_gru_5_bias_m_read_readvariableop8savev2_adam_embedding_5_embeddings_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop.savev2_adam_gru_5_kernel_v_read_readvariableop8savev2_adam_gru_5_recurrent_kernel_v_read_readvariableop,savev2_adam_gru_5_bias_v_read_readvariableop"/device:CPU:0*'
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*▐
_input_shapes╠
╔: :
Ущd:d:: : : : : :	dм:	dм:	м: : :
Ущd:d::	dм:	dм:	м:
Ущd:d::	dм:	dм:	м: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : :
 : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : 
ЮУ
┘
<__inference___backward_cudnn_gru_with_fallback_209340_209479
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         щd`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╡
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:щ         dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:╕
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:щ         dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         d·
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:щ         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:В
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:щ         d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:╤
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         щdu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         щdЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_3c4c0838-5707-4c61-9ea1-6433c1f0c280*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_209478*▓
_input_shapesа
Э:         d:         щd:         d: :щ         d:::::         d:::: ::щ         d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
У*
░
H__inference_sequential_5_layer_call_and_return_conditional_losses_209063

inputs=
9embedding_5_embedding_lookup_read_readvariableop_resource(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3(
$gru_5_statefulpartitionedcall_args_4*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identityИвdense_8/BiasAdd/ReadVariableOpвdense_8/MatMul/ReadVariableOpвembedding_5/embedding_lookupв0embedding_5/embedding_lookup/Read/ReadVariableOpвgru_5/StatefulPartitionedCallb
embedding_5/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         щ┌
0embedding_5/embedding_lookup/Read/ReadVariableOpReadVariableOp9embedding_5_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
УщdЦ
%embedding_5/embedding_lookup/IdentityIdentity8embedding_5/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
УщdЁ
embedding_5/embedding_lookupResourceGather9embedding_5_embedding_lookup_read_readvariableop_resourceembedding_5/Cast:y:01^embedding_5/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*C
_class9
75loc:@embedding_5/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         щdД
'embedding_5/embedding_lookup/Identity_1Identity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*C
_class9
75loc:@embedding_5/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         щdЬ
'embedding_5/embedding_lookup/Identity_2Identity0embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         щdk
gru_5/ShapeShape0embedding_5/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:c
gru_5/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:e
gru_5/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:e
gru_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:я
gru_5/strided_sliceStridedSlicegru_5/Shape:output:0"gru_5/strided_slice/stack:output:0$gru_5/strided_slice/stack_1:output:0$gru_5/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: S
gru_5/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: q
gru_5/zeros/mulMulgru_5/strided_slice:output:0gru_5/zeros/mul/y:output:0*
T0*
_output_shapes
: U
gru_5/zeros/Less/yConst*
value
B :ш*
dtype0*
_output_shapes
: k
gru_5/zeros/LessLessgru_5/zeros/mul:z:0gru_5/zeros/Less/y:output:0*
T0*
_output_shapes
: V
gru_5/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: Е
gru_5/zeros/packedPackgru_5/strided_slice:output:0gru_5/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:V
gru_5/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ~
gru_5/zerosFillgru_5/zeros/packed:output:0gru_5/zeros/Const:output:0*
T0*'
_output_shapes
:         dЇ
gru_5/StatefulPartitionedCallStatefulPartitionedCall0embedding_5/embedding_lookup/Identity_2:output:0gru_5/zeros:output:0$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3$gru_5_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-208826*(
f#R!
__inference_standard_gru_208825*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin	
2*T
_output_shapesB
@:         d:         щd:         d: ▓
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:dЩ
dense_8/MatMulMatMul&gru_5/StatefulPartitionedCall:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ░
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:О
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         О
IdentityIdentitydense_8/Sigmoid:y:0^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^embedding_5/embedding_lookup1^embedding_5/embedding_lookup/Read/ReadVariableOp^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2d
0embedding_5/embedding_lookup/Read/ReadVariableOp0embedding_5/embedding_lookup/Read/ReadVariableOp2<
embedding_5/embedding_lookupembedding_5/embedding_lookup: : : : :& "
 
_user_specified_nameinputs: : 
╕
р
H__inference_sequential_5_layer_call_and_return_conditional_losses_208547
embedding_5_input.
*embedding_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_1(
$gru_5_statefulpartitionedcall_args_2(
$gru_5_statefulpartitionedcall_args_3*
&dense_8_statefulpartitionedcall_args_1*
&dense_8_statefulpartitionedcall_args_2
identityИвdense_8/StatefulPartitionedCallв#embedding_5/StatefulPartitionedCallвgru_5/StatefulPartitionedCall∙
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallembedding_5_input*embedding_5_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-207665*P
fKRI
G__inference_embedding_5_layer_call_and_return_conditional_losses_207659*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*,
_output_shapes
:         щd╦
gru_5/StatefulPartitionedCallStatefulPartitionedCall,embedding_5/StatefulPartitionedCall:output:0$gru_5_statefulpartitionedcall_args_1$gru_5_statefulpartitionedcall_args_2$gru_5_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208498*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208084*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dж
dense_8/StatefulPartitionedCallStatefulPartitionedCall&gru_5/StatefulPartitionedCall:output:0&dense_8_statefulpartitionedcall_args_1&dense_8_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-208535*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_208529*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         ╪
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall^gru_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::2>
gru_5/StatefulPartitionedCallgru_5/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
щ>
╢
(__forward_cudnn_gru_with_fallback_210771

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_210633_210772*<
api_implements*(gru_2a021932-19b6-4b8b-972b-6d94dfd4e5b3*
api_preferred_deviceGPU*
_input_shapes 20
Reshape/ReadVariableOpReshape/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
д0
Е
*__inference_cudnn_gru_with_fallback_207495

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_327f3b28-e868-4060-9d14-c5b7bfead4f0*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
╓J
ё
__inference_standard_gru_210952

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_210845*
_num_original_outputs*
bodyR
while_body_210846*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╓
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  dо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_55dd13e9-718a-4cf5-a8ce-218843da5b97*
api_preferred_deviceCPU*R
_input_shapesA
?:                  d:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
▐+
├
while_body_207300
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
щ>
╢
(__forward_cudnn_gru_with_fallback_207214

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_7_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0п
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0Д
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0"
concatconcat_0:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0*X
backward_function_name><__inference___backward_cudnn_gru_with_fallback_207076_207215*<
api_implements*(gru_7563a40a-6cc7-4cfd-8257-ba857808c14f*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
д0
Е
*__inference_cudnn_gru_with_fallback_209339

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_3c4c0838-5707-4c61-9ea1-6433c1f0c280*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
ў
Л
while_cond_210009
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
╩
Л
while_cond_206879
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
юУ
┘
<__inference___backward_cudnn_gru_with_fallback_207496_207635
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Ив(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         dm
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  d`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         dO
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:╜
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dЬ
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dБ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dВ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:К
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :╕┘Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  du
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:РN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:РN*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_6Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_7Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_8Const*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_9Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_10Const*
valueB:d*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_11Const*
valueB:d*
dtype0*
_output_shapes
:Ш
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:РNъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:РNщ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dщ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:г
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:е
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:б
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:г
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dЬ
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:╖
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:╖
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:╖
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:╖
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:╖
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddЬ
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:╖
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddО
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:╪Е
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dмЛ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dмm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:в
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	мж
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЯ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dС

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмУ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dмФ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	м"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_327f3b28-e868-4060-9d14-c5b7bfead4f0*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_207634*╩
_input_shapes╕
╡:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :╕┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
д0
Е
*__inference_cudnn_gru_with_fallback_207942

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_dd2c674e-9182-4891-8d78-cfc7b4d6e4a7*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
┴	
╔
-__inference_sequential_5_layer_call_fn_208615
embedding_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallembedding_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-208606*Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_208605*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
	2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*?
_input_shapes.
,:         щ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_nameembedding_5_input: : 
╩
Л
while_cond_210845
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
unstack
	unknown_0
	unstack_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м: : : : : :  : : :	 : :
 
▐+
├
while_body_207745
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
й
╦
&__inference_gru_5_layer_call_fn_210365

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-208507*J
fERC
A__inference_gru_5_layer_call_and_return_conditional_losses_208495*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         dВ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         щd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
д0
Е
*__inference_cudnn_gru_with_fallback_207075

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_7563a40a-6cc7-4cfd-8257-ba857808c14f*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
жJ
ё
__inference_standard_gru_207853

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_207744*
_num_original_outputs*
bodyR
while_body_207745*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_dd2c674e-9182-4891-8d78-cfc7b4d6e4a7*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
┘
й
(__inference_dense_8_layer_call_fn_211217

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-208535*L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_208529*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
жJ
ё
__inference_standard_gru_209250

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpвwhileВ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	мa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:м:мc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:щ         dB
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Я
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dЛ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dм|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЧ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dS
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dn
TensorArrayV2_1/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:г
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: [
while/maximum_iterationsConst*
value
B :щ*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :м: :м*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_209141*
_num_original_outputs*
bodyR
while_body_209142*E
_output_shapes3
1: : : : :         d: : : :м: :мK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         dM
while/Identity_5Identitywhile:output:5*
T0*
_output_shapes
: M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: R
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes	
:мM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:мБ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:щ         dh
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         de
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         щd[
runtimeConst"/device:CPU:0*
valueB
 *  А?*
dtype0*
_output_shapes
: л
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dй

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         щdо

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dФ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_3c4c0838-5707-4c61-9ea1-6433c1f0c280*
api_preferred_deviceCPU*J
_input_shapes9
7:         щd:         d:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
д0
Е
*__inference_cudnn_gru_with_fallback_205833

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ИвCudnnRNNвReshape/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_spliti
Reshape/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0D
Reshape/shapeConst*
valueB:
         *
dtype0S
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0X
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0@
	Reshape_1Reshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0@
	Reshape_2Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_3Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0N
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0@
	Reshape_4Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_5Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0@
	Reshape_6Reshapetranspose_6:y:0Const_3:output:0*
T0A
	Reshape_7Reshapesplit_2:output:1Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:5Const_3:output:0*
T05
concat/axisConst*
value	B : *
dtype0н
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0А
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*
rnn_modegruJ
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_7/permConst*!
valueB"          *
dtype0O
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0╞
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Й
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Д

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Н

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Е

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62d9dea7-8d5e-4d01-b597-d07fc355c364*
api_preferred_deviceGPU*
_input_shapes 20
split_1/ReadVariableOpsplit_1/ReadVariableOp20
Reshape/ReadVariableOpReshape/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp:$ 

_user_specified_namebias:&"
 
_user_specified_nameinit_h:& "
 
_user_specified_nameinputs:0,
*
_user_specified_namerecurrent_kernel:&"
 
_user_specified_namekernel
▐+
├
while_body_210437
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0
biasadd_unstack_0&
"matmul_1_readvariableop_resource_0
biasadd_1_unstack_0
identity

identity_1

identity_2

identity_3

identity_4
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource
biasadd_unstack$
 matmul_1_readvariableop_resource
biasadd_1_unstackИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpВ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:О
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dе
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмО
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         мG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: г
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dй
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dмu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         мp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         мI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: й
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         d`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:         dM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         db
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:         dQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         d]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:         dY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:         dI
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:         dZ
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:         dJ
sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:         dQ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:         dV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:         dН
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
element_dtype0*
_output_shapes
: I
add_4/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: I
add_5/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: r
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Г

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Я

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Е

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         d"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"(
biasadd_1_unstackbiasadd_1_unstack_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
biasadd_unstackbiasadd_unstack_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::м::м22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultл
P
embedding_5_input;
#serving_default_embedding_5_input:0         щ;
dense_80
StatefulPartitionedCall:0         tensorflow/serving/predict:├н
┘"
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*\&call_and_return_all_conditional_losses
]__call__
^_default_save_signature"М 
_tf_keras_sequentialэ{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_5", "layers": [{"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}, {"class_name": "GRU", "config": {"name": "gru_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}, {"class_name": "GRU", "config": {"name": "gru_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╣
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"к
_tf_keras_layerР{"class_name": "InputLayer", "name": "embedding_5_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 233], "config": {"batch_input_shape": [null, 233], "dtype": "float32", "sparse": false, "name": "embedding_5_input"}}
т

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"├
_tf_keras_layerй{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 233], "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}
К

cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"с
_tf_keras_layer╟{"class_name": "GRU", "name": "gru_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 100], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
Ї

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
┐
 iter

!beta_1

"beta_2
	#decay
$learning_ratemPmQmR%mS&mT'mUvVvWvX%vY&vZ'v["
	optimizer
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
╖
(non_trainable_variables
trainable_variables
regularization_losses
	variables

)layers
*layer_regularization_losses
+metrics
]__call__
^_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
,non_trainable_variables
trainable_variables
regularization_losses
	variables

-layers
.layer_regularization_losses
/metrics
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
*:(
Ущd2embedding_5/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
Ъ
0non_trainable_variables
trainable_variables
regularization_losses
	variables

1layers
2layer_regularization_losses
3metrics
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ы

%kernel
&recurrent_kernel
'bias
4trainable_variables
5regularization_losses
6	variables
7	keras_api
*h&call_and_return_all_conditional_losses
i__call__"░
_tf_keras_layerЦ{"class_name": "GRUCell", "name": "gru_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell_5", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
Ъ
8non_trainable_variables
trainable_variables
regularization_losses
	variables

9layers
:layer_regularization_losses
;metrics
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :d2dense_8/kernel
:2dense_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ
<non_trainable_variables
trainable_variables
regularization_losses
	variables

=layers
>layer_regularization_losses
?metrics
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:	dм2gru_5/kernel
):'	dм2gru_5/recurrent_kernel
:	м2
gru_5/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
Ъ
Anon_trainable_variables
4trainable_variables
5regularization_losses
6	variables

Blayers
Clayer_regularization_losses
Dmetrics
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	Etotal
	Fcount
G
_fn_kwargs
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
*j&call_and_return_all_conditional_losses
k__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
Ъ
Lnon_trainable_variables
Htrainable_variables
Iregularization_losses
J	variables

Mlayers
Nlayer_regularization_losses
Ometrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/:-
Ущd2Adam/embedding_5/embeddings/m
%:#d2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
$:"	dм2Adam/gru_5/kernel/m
.:,	dм2Adam/gru_5/recurrent_kernel/m
": 	м2Adam/gru_5/bias/m
/:-
Ущd2Adam/embedding_5/embeddings/v
%:#d2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
$:"	dм2Adam/gru_5/kernel/v
.:,	dм2Adam/gru_5/recurrent_kernel/v
": 	м2Adam/gru_5/bias/v
ю2ы
H__inference_sequential_5_layer_call_and_return_conditional_losses_209488
H__inference_sequential_5_layer_call_and_return_conditional_losses_209063
H__inference_sequential_5_layer_call_and_return_conditional_losses_208547
H__inference_sequential_5_layer_call_and_return_conditional_losses_208562└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
-__inference_sequential_5_layer_call_fn_208588
-__inference_sequential_5_layer_call_fn_209499
-__inference_sequential_5_layer_call_fn_208615
-__inference_sequential_5_layer_call_fn_209510└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
!__inference__wrapped_model_205982┴
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *1в.
,К)
embedding_5_input         щ
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ё2ю
G__inference_embedding_5_layer_call_and_return_conditional_losses_209521в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_5_layer_call_fn_209527в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ч2ф
A__inference_gru_5_layer_call_and_return_conditional_losses_210349
A__inference_gru_5_layer_call_and_return_conditional_losses_210774
A__inference_gru_5_layer_call_and_return_conditional_losses_209938
A__inference_gru_5_layer_call_and_return_conditional_losses_211183╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
√2°
&__inference_gru_5_layer_call_fn_210365
&__inference_gru_5_layer_call_fn_211199
&__inference_gru_5_layer_call_fn_210357
&__inference_gru_5_layer_call_fn_211191╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
C__inference_dense_8_layer_call_and_return_conditional_losses_211210в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_8_layer_call_fn_211217в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
=B;
$__inference_signature_wrapper_208636embedding_5_input
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ╡
H__inference_sequential_5_layer_call_and_return_conditional_losses_209488i%&'8в5
.в+
!К
inputs         щ
p 

 
к "%в"
К
0         
Ъ ╡
H__inference_sequential_5_layer_call_and_return_conditional_losses_209063i%&'8в5
.в+
!К
inputs         щ
p

 
к "%в"
К
0         
Ъ └
H__inference_sequential_5_layer_call_and_return_conditional_losses_208547t%&'Cв@
9в6
,К)
embedding_5_input         щ
p

 
к "%в"
К
0         
Ъ └
H__inference_sequential_5_layer_call_and_return_conditional_losses_208562t%&'Cв@
9в6
,К)
embedding_5_input         щ
p 

 
к "%в"
К
0         
Ъ Ш
-__inference_sequential_5_layer_call_fn_208588g%&'Cв@
9в6
,К)
embedding_5_input         щ
p

 
к "К         Н
-__inference_sequential_5_layer_call_fn_209499\%&'8в5
.в+
!К
inputs         щ
p

 
к "К         Ш
-__inference_sequential_5_layer_call_fn_208615g%&'Cв@
9в6
,К)
embedding_5_input         щ
p 

 
к "К         Н
-__inference_sequential_5_layer_call_fn_209510\%&'8в5
.в+
!К
inputs         щ
p 

 
к "К         Э
!__inference__wrapped_model_205982x%&';в8
1в.
,К)
embedding_5_input         щ
к "1к.
,
dense_8!К
dense_8         м
G__inference_embedding_5_layer_call_and_return_conditional_losses_209521a0в-
&в#
!К
inputs         щ
к "*в'
 К
0         щd
Ъ Д
,__inference_embedding_5_layer_call_fn_209527T0в-
&в#
!К
inputs         щ
к "К         щd│
A__inference_gru_5_layer_call_and_return_conditional_losses_210349n%&'@в=
6в3
%К"
inputs         щd

 
p 

 
к "%в"
К
0         d
Ъ ┬
A__inference_gru_5_layer_call_and_return_conditional_losses_210774}%&'OвL
EвB
4Ъ1
/К,
inputs/0                  d

 
p

 
к "%в"
К
0         d
Ъ │
A__inference_gru_5_layer_call_and_return_conditional_losses_209938n%&'@в=
6в3
%К"
inputs         щd

 
p

 
к "%в"
К
0         d
Ъ ┬
A__inference_gru_5_layer_call_and_return_conditional_losses_211183}%&'OвL
EвB
4Ъ1
/К,
inputs/0                  d

 
p 

 
к "%в"
К
0         d
Ъ Л
&__inference_gru_5_layer_call_fn_210365a%&'@в=
6в3
%К"
inputs         щd

 
p 

 
к "К         dЪ
&__inference_gru_5_layer_call_fn_211199p%&'OвL
EвB
4Ъ1
/К,
inputs/0                  d

 
p 

 
к "К         dЛ
&__inference_gru_5_layer_call_fn_210357a%&'@в=
6в3
%К"
inputs         щd

 
p

 
к "К         dЪ
&__inference_gru_5_layer_call_fn_211191p%&'OвL
EвB
4Ъ1
/К,
inputs/0                  d

 
p

 
к "К         dг
C__inference_dense_8_layer_call_and_return_conditional_losses_211210\/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ {
(__inference_dense_8_layer_call_fn_211217O/в,
%в"
 К
inputs         d
к "К         ╢
$__inference_signature_wrapper_208636Н%&'PвM
в 
FкC
A
embedding_5_input,К)
embedding_5_input         щ"1к.
,
dense_8!К
dense_8         