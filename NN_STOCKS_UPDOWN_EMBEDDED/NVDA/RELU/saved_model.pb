ю╝)
Ў§
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
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02unknown8▓В'
ї
embedding_20/embeddingsVarHandleOp*
shape:
Њжd*(
shared_nameembedding_20/embeddings*
dtype0*
_output_shapes
: 
Ё
+embedding_20/embeddings/Read/ReadVariableOpReadVariableOpembedding_20/embeddings*
dtype0* 
_output_shapes
:
Њжd
z
dense_30/kernelVarHandleOp*
shape
:dd* 
shared_namedense_30/kernel*
dtype0*
_output_shapes
: 
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
dtype0*
_output_shapes

:dd
r
dense_30/biasVarHandleOp*
shape:d*
shared_namedense_30/bias*
dtype0*
_output_shapes
: 
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
dtype0*
_output_shapes
:d
z
dense_31/kernelVarHandleOp*
shape
:d* 
shared_namedense_31/kernel*
dtype0*
_output_shapes
: 
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
dtype0*
_output_shapes

:d
r
dense_31/biasVarHandleOp*
shape:*
shared_namedense_31/bias*
dtype0*
_output_shapes
: 
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
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
w
gru_20/kernelVarHandleOp*
shape:	dг*
shared_namegru_20/kernel*
dtype0*
_output_shapes
: 
p
!gru_20/kernel/Read/ReadVariableOpReadVariableOpgru_20/kernel*
dtype0*
_output_shapes
:	dг
І
gru_20/recurrent_kernelVarHandleOp*
shape:	dг*(
shared_namegru_20/recurrent_kernel*
dtype0*
_output_shapes
: 
ё
+gru_20/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_20/recurrent_kernel*
dtype0*
_output_shapes
:	dг
s
gru_20/biasVarHandleOp*
shape:	г*
shared_namegru_20/bias*
dtype0*
_output_shapes
: 
l
gru_20/bias/Read/ReadVariableOpReadVariableOpgru_20/bias*
dtype0*
_output_shapes
:	г
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
џ
Adam/embedding_20/embeddings/mVarHandleOp*
shape:
Њжd*/
shared_name Adam/embedding_20/embeddings/m*
dtype0*
_output_shapes
: 
Њ
2Adam/embedding_20/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/m*
dtype0* 
_output_shapes
:
Њжd
ѕ
Adam/dense_30/kernel/mVarHandleOp*
shape
:dd*'
shared_nameAdam/dense_30/kernel/m*
dtype0*
_output_shapes
: 
Ђ
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
dtype0*
_output_shapes

:dd
ђ
Adam/dense_30/bias/mVarHandleOp*
shape:d*%
shared_nameAdam/dense_30/bias/m*
dtype0*
_output_shapes
: 
y
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
dtype0*
_output_shapes
:d
ѕ
Adam/dense_31/kernel/mVarHandleOp*
shape
:d*'
shared_nameAdam/dense_31/kernel/m*
dtype0*
_output_shapes
: 
Ђ
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m*
dtype0*
_output_shapes

:d
ђ
Adam/dense_31/bias/mVarHandleOp*
shape:*%
shared_nameAdam/dense_31/bias/m*
dtype0*
_output_shapes
: 
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
dtype0*
_output_shapes
:
Ё
Adam/gru_20/kernel/mVarHandleOp*
shape:	dг*%
shared_nameAdam/gru_20/kernel/m*
dtype0*
_output_shapes
: 
~
(Adam/gru_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_20/kernel/m*
dtype0*
_output_shapes
:	dг
Ў
Adam/gru_20/recurrent_kernel/mVarHandleOp*
shape:	dг*/
shared_name Adam/gru_20/recurrent_kernel/m*
dtype0*
_output_shapes
: 
њ
2Adam/gru_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_20/recurrent_kernel/m*
dtype0*
_output_shapes
:	dг
Ђ
Adam/gru_20/bias/mVarHandleOp*
shape:	г*#
shared_nameAdam/gru_20/bias/m*
dtype0*
_output_shapes
: 
z
&Adam/gru_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_20/bias/m*
dtype0*
_output_shapes
:	г
џ
Adam/embedding_20/embeddings/vVarHandleOp*
shape:
Њжd*/
shared_name Adam/embedding_20/embeddings/v*
dtype0*
_output_shapes
: 
Њ
2Adam/embedding_20/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/v*
dtype0* 
_output_shapes
:
Њжd
ѕ
Adam/dense_30/kernel/vVarHandleOp*
shape
:dd*'
shared_nameAdam/dense_30/kernel/v*
dtype0*
_output_shapes
: 
Ђ
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
dtype0*
_output_shapes

:dd
ђ
Adam/dense_30/bias/vVarHandleOp*
shape:d*%
shared_nameAdam/dense_30/bias/v*
dtype0*
_output_shapes
: 
y
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
dtype0*
_output_shapes
:d
ѕ
Adam/dense_31/kernel/vVarHandleOp*
shape
:d*'
shared_nameAdam/dense_31/kernel/v*
dtype0*
_output_shapes
: 
Ђ
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v*
dtype0*
_output_shapes

:d
ђ
Adam/dense_31/bias/vVarHandleOp*
shape:*%
shared_nameAdam/dense_31/bias/v*
dtype0*
_output_shapes
: 
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
dtype0*
_output_shapes
:
Ё
Adam/gru_20/kernel/vVarHandleOp*
shape:	dг*%
shared_nameAdam/gru_20/kernel/v*
dtype0*
_output_shapes
: 
~
(Adam/gru_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_20/kernel/v*
dtype0*
_output_shapes
:	dг
Ў
Adam/gru_20/recurrent_kernel/vVarHandleOp*
shape:	dг*/
shared_name Adam/gru_20/recurrent_kernel/v*
dtype0*
_output_shapes
: 
њ
2Adam/gru_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_20/recurrent_kernel/v*
dtype0*
_output_shapes
:	dг
Ђ
Adam/gru_20/bias/vVarHandleOp*
shape:	г*#
shared_nameAdam/gru_20/bias/v*
dtype0*
_output_shapes
: 
z
&Adam/gru_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_20/bias/v*
dtype0*
_output_shapes
:	г

NoOpNoOp
т1
ConstConst"/device:CPU:0*а1
valueќ1BЊ1 Bї1
џ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
л
'iter

(beta_1

)beta_2
	*decay
+learning_ratem[m\m]!m^"m_,m`-ma.mbvcvdve!vf"vg,vh-vi.vj
8
0
,1
-2
.3
4
5
!6
"7
 
8
0
,1
-2
.3
4
5
!6
"7
џ
/non_trainable_variables
trainable_variables
regularization_losses
		variables

0layers
1layer_regularization_losses
2metrics
 
 
 
 
џ
3non_trainable_variables
trainable_variables
regularization_losses
	variables

4layers
5layer_regularization_losses
6metrics
ge
VARIABLE_VALUEembedding_20/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
џ
7non_trainable_variables
trainable_variables
regularization_losses
	variables

8layers
9layer_regularization_losses
:metrics
~

,kernel
-recurrent_kernel
.bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
 

,0
-1
.2
 

,0
-1
.2
џ
?non_trainable_variables
trainable_variables
regularization_losses
	variables

@layers
Alayer_regularization_losses
Bmetrics
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
џ
Cnon_trainable_variables
trainable_variables
regularization_losses
	variables

Dlayers
Elayer_regularization_losses
Fmetrics
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
џ
Gnon_trainable_variables
#trainable_variables
$regularization_losses
%	variables

Hlayers
Ilayer_regularization_losses
Jmetrics
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
SQ
VARIABLE_VALUEgru_20/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_20/recurrent_kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEgru_20/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 

K0
 
 
 
 
 
 
 
 

,0
-1
.2
 

,0
-1
.2
џ
Lnon_trainable_variables
;trainable_variables
<regularization_losses
=	variables

Mlayers
Nlayer_regularization_losses
Ometrics
 

0
 
 
 
 
 
 
 
 
 
 
x
	Ptotal
	Qcount
R
_fn_kwargs
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
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
P0
Q1
џ
Wnon_trainable_variables
Strainable_variables
Tregularization_losses
U	variables

Xlayers
Ylayer_regularization_losses
Zmetrics

P0
Q1
 
 
 
Іѕ
VARIABLE_VALUEAdam/embedding_20/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_20/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/gru_20/recurrent_kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_20/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUEAdam/embedding_20/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_20/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/gru_20/recurrent_kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/gru_20/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Є
"serving_default_embedding_20_inputPlaceholder*
shape:         ж*
dtype0*(
_output_shapes
:         ж
┐
StatefulPartitionedCallStatefulPartitionedCall"serving_default_embedding_20_inputembedding_20/embeddingsgru_20/kernelgru_20/recurrent_kernelgru_20/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias*-
_gradient_op_typePartitionedCall-827834*-
f(R&
$__inference_signature_wrapper_825151*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
Ђ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+embedding_20/embeddings/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!gru_20/kernel/Read/ReadVariableOp+gru_20/recurrent_kernel/Read/ReadVariableOpgru_20/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/embedding_20/embeddings/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp(Adam/gru_20/kernel/m/Read/ReadVariableOp2Adam/gru_20/recurrent_kernel/m/Read/ReadVariableOp&Adam/gru_20/bias/m/Read/ReadVariableOp2Adam/embedding_20/embeddings/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp(Adam/gru_20/kernel/v/Read/ReadVariableOp2Adam/gru_20/recurrent_kernel/v/Read/ReadVariableOp&Adam/gru_20/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-827887*(
f#R!
__inference__traced_save_827886*
Tout
2*-
config_proto

CPU

GPU2*0J 8*,
Tin%
#2!	*
_output_shapes
: 
љ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_20/embeddingsdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_20/kernelgru_20/recurrent_kernelgru_20/biastotalcountAdam/embedding_20/embeddings/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/gru_20/kernel/mAdam/gru_20/recurrent_kernel/mAdam/gru_20/bias/mAdam/embedding_20/embeddings/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/gru_20/kernel/vAdam/gru_20/recurrent_kernel/vAdam/gru_20/bias/v*-
_gradient_op_typePartitionedCall-827993*+
f&R$
"__inference__traced_restore_827992*
Tout
2*-
config_proto

CPU

GPU2*0J 8*+
Tin$
"2 *
_output_shapes
: йт&
¤	
П
D__inference_dense_31_layer_call_and_return_conditional_losses_827761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
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
:         ё
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
э
І
while_cond_824216
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
щ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_824967

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
:         dг
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-824737*(
f#R!
__inference_standard_gru_824736*
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
@:         d:         жd:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
Ф
╠
'__inference_gru_20_layer_call_fn_827724

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824970*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824556*
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
:         dѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
я+
├
while_body_822101
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
оJ
ы
__inference_standard_gru_826238

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_826131*
_num_original_outputs*
bodyR
while_body_826132*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:о
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
:Є
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
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  d«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_aef7fe5d-248b-4091-8fb3-e3f680a79f46*
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
оJ
ы
__inference_standard_gru_823458

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_823351*
_num_original_outputs*
bodyR
while_body_823352*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:о
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
:Є
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
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  d«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_64add08a-754e-4b31-9a7c-25dc3047afac*
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
└

Є
.__inference_sequential_20_layer_call_fn_826043

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-825117*R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_825116*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
Љ
ж
B__inference_gru_20_layer_call_and_return_conditional_losses_826878
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall=
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
:Л
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
B :У*
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
:         dХ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-826648*(
f#R!
__inference_standard_gru_826647*
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
H:         d:                  d:         d: ѓ
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
П3
┌
I__inference_sequential_20_layer_call_and_return_conditional_losses_826017

inputs>
:embedding_20_embedding_lookup_read_readvariableop_resource)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3)
%gru_20_statefulpartitionedcall_args_4+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identityѕбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpбdense_31/BiasAdd/ReadVariableOpбdense_31/MatMul/ReadVariableOpбembedding_20/embedding_lookupб1embedding_20/embedding_lookup/Read/ReadVariableOpбgru_20/StatefulPartitionedCallc
embedding_20/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         ж▄
1embedding_20/embedding_lookup/Read/ReadVariableOpReadVariableOp:embedding_20_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Њжdў
&embedding_20/embedding_lookup/IdentityIdentity9embedding_20/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Њжdш
embedding_20/embedding_lookupResourceGather:embedding_20_embedding_lookup_read_readvariableop_resourceembedding_20/Cast:y:02^embedding_20/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@embedding_20/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         жdЄ
(embedding_20/embedding_lookup/Identity_1Identity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@embedding_20/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         жdъ
(embedding_20/embedding_lookup/Identity_2Identity1embedding_20/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         жdm
gru_20/ShapeShape1embedding_20/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:d
gru_20/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
gru_20/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
gru_20/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
gru_20/strided_sliceStridedSlicegru_20/Shape:output:0#gru_20/strided_slice/stack:output:0%gru_20/strided_slice/stack_1:output:0%gru_20/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: T
gru_20/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: t
gru_20/zeros/mulMulgru_20/strided_slice:output:0gru_20/zeros/mul/y:output:0*
T0*
_output_shapes
: V
gru_20/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
gru_20/zeros/LessLessgru_20/zeros/mul:z:0gru_20/zeros/Less/y:output:0*
T0*
_output_shapes
: W
gru_20/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: ѕ
gru_20/zeros/packedPackgru_20/strided_slice:output:0gru_20/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
gru_20/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
gru_20/zerosFillgru_20/zeros/packed:output:0gru_20/zeros/Const:output:0*
T0*'
_output_shapes
:         dЩ
gru_20/StatefulPartitionedCallStatefulPartitionedCall1embedding_20/embedding_lookup/Identity_2:output:0gru_20/zeros:output:0%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3%gru_20_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-825773*(
f#R!
__inference_standard_gru_825772*
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
@:         d:         жd:         d: ┤
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:ddю
dense_30/MatMulMatMul'gru_20/StatefulPartitionedCall:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d▓
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:dЉ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         d┤
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:dљ
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Љ
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:         О
IdentityIdentitydense_31/Sigmoid:y:0 ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp^embedding_20/embedding_lookup2^embedding_20/embedding_lookup/Read/ReadVariableOp^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2f
1embedding_20/embedding_lookup/Read/ReadVariableOp1embedding_20/embedding_lookup/Read/ReadVariableOp2>
embedding_20/embedding_lookupembedding_20/embedding_lookup2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
╩
І
while_cond_823351
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
Ф
╠
'__inference_gru_20_layer_call_fn_827732

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824979*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824967*
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
:         dѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
я+
├
while_body_823352
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ЬЊ
┘
<__inference___backward_cudnn_gru_with_fallback_826328_826467
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
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
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dѓ
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
:і
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :И┘ќ
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_aef7fe5d-248b-4091-8fb3-e3f680a79f46*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_826466*╩
_input_shapesИ
х:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
ц0
Ё
*__inference_cudnn_gru_with_fallback_823547

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_64add08a-754e-4b31-9a7c-25dc3047afac*
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
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_827575_827714
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_9b7933c6-ea4d-4110-9d29-cb417a74bedf*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_827713*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
▒
н
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131

inputs1
-embedding_lookup_read_readvariableop_resource
identityѕбembedding_lookupб$embedding_lookup/Read/ReadVariableOpV
CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         ж┬
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Њжd~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Њжd┤
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceCast:y:0%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         жdЯ
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         жdё
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         жdФ
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*,
_output_shapes
:         жd"
identityIdentity:output:0*+
_input_shapes
:         ж:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
ж>
Х
(__forward_cudnn_gru_with_fallback_826466

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_826328_826467*<
api_implements*(gru_aef7fe5d-248b-4091-8fb3-e3f680a79f46*
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
дJ
ы
__inference_standard_gru_824736

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_824627*
_num_original_outputs*
bodyR
while_body_824628*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_40cb0202-b7f7-4b69-b739-bec6de949ed9*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_827164_827303
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_5031afeb-0180-467a-9e22-ae53433c2df5*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_827302*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
ж>
Х
(__forward_cudnn_gru_with_fallback_827713

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_827575_827714*<
api_implements*(gru_9b7933c6-ea4d-4110-9d29-cb417a74bedf*
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
я+
├
while_body_826966
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
╩
І
while_cond_823771
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
ж>
Х
(__forward_cudnn_gru_with_fallback_823686

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_823548_823687*<
api_implements*(gru_64add08a-754e-4b31-9a7c-25dc3047afac*
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
ц0
Ё
*__inference_cudnn_gru_with_fallback_826736

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_92e3f741-6ed6-4cb8-993a-a8b40f6d89e0*
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
І
while_cond_826540
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
э
І
while_cond_822100
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
▓

Ѕ
$__inference_signature_wrapper_825151
embedding_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallembedding_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-825140**
f%R#
!__inference__wrapped_model_822454*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
█
ф
)__inference_dense_31_layer_call_fn_827768

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825035*M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_825029*
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
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
ц0
Ё
*__inference_cudnn_gru_with_fallback_827574

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_9b7933c6-ea4d-4110-9d29-cb417a74bedf*
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
ЬЊ
┘
<__inference___backward_cudnn_gru_with_fallback_826737_826876
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
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
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dѓ
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
:і
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :И┘ќ
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_92e3f741-6ed6-4cb8-993a-a8b40f6d89e0*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_826875*╩
_input_shapesИ
х:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
я+
├
while_body_824628
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ц0
Ё
*__inference_cudnn_gru_with_fallback_824414

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62ff8d29-bd92-402b-a51a-a46bbf23f9d4*
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
ж>
Х
(__forward_cudnn_gru_with_fallback_826000

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_825862_826001*<
api_implements*(gru_fe3a0d69-7df0-451c-97b0-9678755fe9bd*
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
я+
├
while_body_824217
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ж>
Х
(__forward_cudnn_gru_with_fallback_824964

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_824826_824965*<
api_implements*(gru_40cb0202-b7f7-4b69-b739-bec6de949ed9*
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
└

Є
.__inference_sequential_20_layer_call_fn_826030

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-825085*R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_825084*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
дJ
ы
__inference_standard_gru_825772

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_825663*
_num_original_outputs*
bodyR
while_body_825664*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_fe3a0d69-7df0-451c-97b0-9678755fe9bd*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
Є
У
I__inference_sequential_20_layer_call_and_return_conditional_losses_825047
embedding_20_input/
+embedding_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3+
'dense_30_statefulpartitionedcall_args_1+
'dense_30_statefulpartitionedcall_args_2+
'dense_31_statefulpartitionedcall_args_1+
'dense_31_statefulpartitionedcall_args_2
identityѕб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб$embedding_20/StatefulPartitionedCallбgru_20/StatefulPartitionedCall§
$embedding_20/StatefulPartitionedCallStatefulPartitionedCallembedding_20_input+embedding_20_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-824137*Q
fLRJ
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131*
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
:         жdЛ
gru_20/StatefulPartitionedCallStatefulPartitionedCall-embedding_20/StatefulPartitionedCall:output:0%gru_20_statefulpartitionedcall_args_1%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824970*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824556*
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
:         dФ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall'gru_20/StatefulPartitionedCall:output:0'dense_30_statefulpartitionedcall_args_1'dense_30_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825007*M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_825001*
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
:         dГ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0'dense_31_statefulpartitionedcall_args_1'dense_31_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825035*M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_825029*
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
:          
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
С

Њ
.__inference_sequential_20_layer_call_fn_825128
embedding_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallembedding_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-825117*R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_825116*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
Є
У
I__inference_sequential_20_layer_call_and_return_conditional_losses_825065
embedding_20_input/
+embedding_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3+
'dense_30_statefulpartitionedcall_args_1+
'dense_30_statefulpartitionedcall_args_2+
'dense_31_statefulpartitionedcall_args_1+
'dense_31_statefulpartitionedcall_args_2
identityѕб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб$embedding_20/StatefulPartitionedCallбgru_20/StatefulPartitionedCall§
$embedding_20/StatefulPartitionedCallStatefulPartitionedCallembedding_20_input+embedding_20_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-824137*Q
fLRJ
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131*
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
:         жdЛ
gru_20/StatefulPartitionedCallStatefulPartitionedCall-embedding_20/StatefulPartitionedCall:output:0%gru_20_statefulpartitionedcall_args_1%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824979*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824967*
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
:         dФ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall'gru_20/StatefulPartitionedCall:output:0'dense_30_statefulpartitionedcall_args_1'dense_30_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825007*M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_825001*
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
:         dГ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0'dense_31_statefulpartitionedcall_args_1'dense_31_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825035*M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_825029*
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
:          
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
ж>
Х
(__forward_cudnn_gru_with_fallback_827302

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_827164_827303*<
api_implements*(gru_5031afeb-0180-467a-9e22-ae53433c2df5*
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
э
І
while_cond_826965
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
ц0
Ё
*__inference_cudnn_gru_with_fallback_826327

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_aef7fe5d-248b-4091-8fb3-e3f680a79f46*
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
мx
╩
"__inference__traced_restore_827992
file_prefix,
(assignvariableop_embedding_20_embeddings&
"assignvariableop_1_dense_30_kernel$
 assignvariableop_2_dense_30_bias&
"assignvariableop_3_dense_31_kernel$
 assignvariableop_4_dense_31_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rate%
!assignvariableop_10_gru_20_kernel/
+assignvariableop_11_gru_20_recurrent_kernel#
assignvariableop_12_gru_20_bias
assignvariableop_13_total
assignvariableop_14_count6
2assignvariableop_15_adam_embedding_20_embeddings_m.
*assignvariableop_16_adam_dense_30_kernel_m,
(assignvariableop_17_adam_dense_30_bias_m.
*assignvariableop_18_adam_dense_31_kernel_m,
(assignvariableop_19_adam_dense_31_bias_m,
(assignvariableop_20_adam_gru_20_kernel_m6
2assignvariableop_21_adam_gru_20_recurrent_kernel_m*
&assignvariableop_22_adam_gru_20_bias_m6
2assignvariableop_23_adam_embedding_20_embeddings_v.
*assignvariableop_24_adam_dense_30_kernel_v,
(assignvariableop_25_adam_dense_30_bias_v.
*assignvariableop_26_adam_dense_31_kernel_v,
(assignvariableop_27_adam_dense_31_bias_v,
(assignvariableop_28_adam_gru_20_kernel_v6
2assignvariableop_29_adam_gru_20_recurrent_kernel_v*
&assignvariableop_30_adam_gru_20_bias_v
identity_32ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1і
RestoreV2/tensor_namesConst"/device:CPU:0*░
valueдBБB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:«
RestoreV2/shape_and_slicesConst"/device:CPU:0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:║
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*-
dtypes#
!2	*љ
_output_shapes~
|:::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:ё
AssignVariableOpAssignVariableOp(assignvariableop_embedding_20_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:ѓ
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_30_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:ђ
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_30_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:ѓ
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_31_kernelIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:ђ
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_31_biasIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0	*
_output_shapes
:|
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0*
dtype0	*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:~
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:}
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:Ё
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Ѓ
AssignVariableOp_10AssignVariableOp!assignvariableop_10_gru_20_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Ї
AssignVariableOp_11AssignVariableOp+assignvariableop_11_gru_20_recurrent_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Ђ
AssignVariableOp_12AssignVariableOpassignvariableop_12_gru_20_biasIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:{
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:ћ
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_embedding_20_embeddings_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:ї
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_30_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:і
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense_30_bias_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:ї
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_31_kernel_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:і
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_31_bias_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:і
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_gru_20_kernel_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:ћ
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_gru_20_recurrent_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:ѕ
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_gru_20_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:ћ
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_embedding_20_embeddings_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:ї
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_30_kernel_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:і
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_30_bias_vIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:ї
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense_31_kernel_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:і
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_dense_31_bias_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:і
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_gru_20_kernel_vIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:ћ
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_gru_20_recurrent_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:ѕ
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_gru_20_bias_vIdentity_30:output:0*
dtype0*
_output_shapes
 ї
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
:х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 щ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: є
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_32Identity_32:output:0*њ
_input_shapesђ
~: :::::::::::::::::::::::::::::::2*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_28AssignVariableOp_28: : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : : : : : :
 
╣
╬
'__inference_gru_20_layer_call_fn_826886
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-823690*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_823689*
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
:         dѓ
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
ц0
Ё
*__inference_cudnn_gru_with_fallback_824825

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_40cb0202-b7f7-4b69-b739-bec6de949ed9*
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
э
І
while_cond_825663
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
щ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_824556

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
:         dг
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-824326*(
f#R!
__inference_standard_gru_824325*
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
@:         d:         жd:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_825862_826001
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_fe3a0d69-7df0-451c-97b0-9678755fe9bd*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_826000*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ц0
Ё
*__inference_cudnn_gru_with_fallback_827163

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_5031afeb-0180-467a-9e22-ae53433c2df5*
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
ж>
Х
(__forward_cudnn_gru_with_fallback_824106

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_823968_824107*<
api_implements*(gru_96187d1d-8adb-4d13-9661-83b3f69b971e*
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
Љ
ж
B__inference_gru_20_layer_call_and_return_conditional_losses_826469
inputs_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall=
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
:Л
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
B :У*
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
:         dХ
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-826239*(
f#R!
__inference_standard_gru_826238*
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
H:         d:                  d:         d: ѓ
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
С

Њ
.__inference_sequential_20_layer_call_fn_825096
embedding_20_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallembedding_20_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*-
_gradient_op_typePartitionedCall-825085*R
fMRK
I__inference_sequential_20_layer_call_and_return_conditional_losses_825084*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
╣
╬
'__inference_gru_20_layer_call_fn_826894
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824110*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824109*
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
:         dѓ
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
дJ
ы
__inference_standard_gru_822209

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_822100*
_num_original_outputs*
bodyR
while_body_822101*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_4fe44b6e-07a2-477f-84a1-5ebeb806b94d*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
ж>
Х
(__forward_cudnn_gru_with_fallback_824553

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_824415_824554*<
api_implements*(gru_62ff8d29-bd92-402b-a51a-a46bbf23f9d4*
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
¤	
П
D__inference_dense_31_layer_call_and_return_conditional_losses_825029

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
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
:         ё
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
дJ
ы
__inference_standard_gru_825340

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_825231*
_num_original_outputs*
bodyR
while_body_825232*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_592ce9d6-8c49-4d71-8a35-0ba3527a5754*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
дJ
ы
__inference_standard_gru_827074

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_826965*
_num_original_outputs*
bodyR
while_body_826966*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_5031afeb-0180-467a-9e22-ae53433c2df5*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
ж>
Х
(__forward_cudnn_gru_with_fallback_822437

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_822299_822438*<
api_implements*(gru_4fe44b6e-07a2-477f-84a1-5ebeb806b94d*
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
дJ
ы
__inference_standard_gru_827485

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_827376*
_num_original_outputs*
bodyR
while_body_827377*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_9b7933c6-ea4d-4110-9d29-cb417a74bedf*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
я+
├
while_body_826132
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_824415_824554
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62ff8d29-bd92-402b-a51a-a46bbf23f9d4*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_824553*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
л	
П
D__inference_dense_30_layer_call_and_return_conditional_losses_827743

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         dІ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*.
_input_shapes
:         d::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ц0
Ё
*__inference_cudnn_gru_with_fallback_822298

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_4fe44b6e-07a2-477f-84a1-5ebeb806b94d*
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
с
▄
I__inference_sequential_20_layer_call_and_return_conditional_losses_825084

inputs/
+embedding_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3+
'dense_30_statefulpartitionedcall_args_1+
'dense_30_statefulpartitionedcall_args_2+
'dense_31_statefulpartitionedcall_args_1+
'dense_31_statefulpartitionedcall_args_2
identityѕб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб$embedding_20/StatefulPartitionedCallбgru_20/StatefulPartitionedCallы
$embedding_20/StatefulPartitionedCallStatefulPartitionedCallinputs+embedding_20_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-824137*Q
fLRJ
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131*
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
:         жdЛ
gru_20/StatefulPartitionedCallStatefulPartitionedCall-embedding_20/StatefulPartitionedCall:output:0%gru_20_statefulpartitionedcall_args_1%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824970*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824556*
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
:         dФ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall'gru_20/StatefulPartitionedCall:output:0'dense_30_statefulpartitionedcall_args_1'dense_30_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825007*M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_825001*
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
:         dГ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0'dense_31_statefulpartitionedcall_args_1'dense_31_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825035*M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_825029*
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
:          
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
с
▄
I__inference_sequential_20_layer_call_and_return_conditional_losses_825116

inputs/
+embedding_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_1)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3+
'dense_30_statefulpartitionedcall_args_1+
'dense_30_statefulpartitionedcall_args_2+
'dense_31_statefulpartitionedcall_args_1+
'dense_31_statefulpartitionedcall_args_2
identityѕб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб$embedding_20/StatefulPartitionedCallбgru_20/StatefulPartitionedCallы
$embedding_20/StatefulPartitionedCallStatefulPartitionedCallinputs+embedding_20_statefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-824137*Q
fLRJ
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131*
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
:         жdЛ
gru_20/StatefulPartitionedCallStatefulPartitionedCall-embedding_20/StatefulPartitionedCall:output:0%gru_20_statefulpartitionedcall_args_1%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3*-
_gradient_op_typePartitionedCall-824979*K
fFRD
B__inference_gru_20_layer_call_and_return_conditional_losses_824967*
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
:         dФ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall'gru_20/StatefulPartitionedCall:output:0'dense_30_statefulpartitionedcall_args_1'dense_30_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825007*M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_825001*
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
:         dГ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0'dense_31_statefulpartitionedcall_args_1'dense_31_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825035*M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_825029*
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
:          
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
П3
┌
I__inference_sequential_20_layer_call_and_return_conditional_losses_825585

inputs>
:embedding_20_embedding_lookup_read_readvariableop_resource)
%gru_20_statefulpartitionedcall_args_2)
%gru_20_statefulpartitionedcall_args_3)
%gru_20_statefulpartitionedcall_args_4+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identityѕбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpбdense_31/BiasAdd/ReadVariableOpбdense_31/MatMul/ReadVariableOpбembedding_20/embedding_lookupб1embedding_20/embedding_lookup/Read/ReadVariableOpбgru_20/StatefulPartitionedCallc
embedding_20/CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         ж▄
1embedding_20/embedding_lookup/Read/ReadVariableOpReadVariableOp:embedding_20_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Њжdў
&embedding_20/embedding_lookup/IdentityIdentity9embedding_20/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Њжdш
embedding_20/embedding_lookupResourceGather:embedding_20_embedding_lookup_read_readvariableop_resourceembedding_20/Cast:y:02^embedding_20/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*D
_class:
86loc:@embedding_20/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         жdЄ
(embedding_20/embedding_lookup/Identity_1Identity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@embedding_20/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         жdъ
(embedding_20/embedding_lookup/Identity_2Identity1embedding_20/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         жdm
gru_20/ShapeShape1embedding_20/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:d
gru_20/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
gru_20/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
gru_20/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
gru_20/strided_sliceStridedSlicegru_20/Shape:output:0#gru_20/strided_slice/stack:output:0%gru_20/strided_slice/stack_1:output:0%gru_20/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: T
gru_20/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: t
gru_20/zeros/mulMulgru_20/strided_slice:output:0gru_20/zeros/mul/y:output:0*
T0*
_output_shapes
: V
gru_20/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
gru_20/zeros/LessLessgru_20/zeros/mul:z:0gru_20/zeros/Less/y:output:0*
T0*
_output_shapes
: W
gru_20/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: ѕ
gru_20/zeros/packedPackgru_20/strided_slice:output:0gru_20/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
gru_20/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
gru_20/zerosFillgru_20/zeros/packed:output:0gru_20/zeros/Const:output:0*
T0*'
_output_shapes
:         dЩ
gru_20/StatefulPartitionedCallStatefulPartitionedCall1embedding_20/embedding_lookup/Identity_2:output:0gru_20/zeros:output:0%gru_20_statefulpartitionedcall_args_2%gru_20_statefulpartitionedcall_args_3%gru_20_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-825341*(
f#R!
__inference_standard_gru_825340*
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
@:         d:         жd:         d: ┤
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:ddю
dense_30/MatMulMatMul'gru_20/StatefulPartitionedCall:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d▓
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:dЉ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         db
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         d┤
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:dљ
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Љ
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:         О
IdentityIdentitydense_31/Sigmoid:y:0 ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp^embedding_20/embedding_lookup2^embedding_20/embedding_lookup/Read/ReadVariableOp^gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2f
1embedding_20/embedding_lookup/Read/ReadVariableOp1embedding_20/embedding_lookup/Read/ReadVariableOp2>
embedding_20/embedding_lookupembedding_20/embedding_lookup2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
gru_20/StatefulPartitionedCallgru_20/StatefulPartitionedCall2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
э
І
while_cond_825231
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
Ъ
і
-__inference_embedding_20_layer_call_fn_826060

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*-
_gradient_op_typePartitionedCall-824137*Q
fLRJ
H__inference_embedding_20_layer_call_and_return_conditional_losses_824131*
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
:         жdЄ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         жd"
identityIdentity:output:0*+
_input_shapes
:         ж:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
э
І
while_cond_827376
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
я+
├
while_body_827377
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_824826_824965
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_40cb0202-b7f7-4b69-b739-bec6de949ed9*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_824964*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :
 : : : : : : :	 : : : : :  : : : : : : : : : : : : 
Ѕ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_824109

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
_gradient_op_typePartitionedCall-823879*(
f#R!
__inference_standard_gru_823878*
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
H:         d:                  d:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
я+
├
while_body_825664
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
█
ф
)__inference_dense_30_layer_call_fn_827750

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-825007*M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_825001*
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
:         dѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*.
_input_shapes
:         d::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
я+
├
while_body_823772
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ц0
Ё
*__inference_cudnn_gru_with_fallback_825861

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_fe3a0d69-7df0-451c-97b0-9678755fe9bd*
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
дJ
ы
__inference_standard_gru_824325

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ж         dB
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
B :ж*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_824216*
_num_original_outputs*
bodyR
while_body_824217*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:╬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*,
_output_shapes
:ж         dh
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
:Є
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
:Ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         жd[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dЕ

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*,
_output_shapes
:         жd«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_62ff8d29-bd92-402b-a51a-a46bbf23f9d4*
api_preferred_deviceCPU*J
_input_shapes9
7:         жd:         d:::22
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
я+
├
while_body_826541
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
л	
П
D__inference_dense_30_layer_call_and_return_conditional_losses_825001

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:ddi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:dv
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         dІ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*.
_input_shapes
:         d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
э
І
while_cond_824627
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
а@
░
__inference__traced_save_827886
file_prefix6
2savev2_embedding_20_embeddings_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_gru_20_kernel_read_readvariableop6
2savev2_gru_20_recurrent_kernel_read_readvariableop*
&savev2_gru_20_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_embedding_20_embeddings_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop3
/savev2_adam_gru_20_kernel_m_read_readvariableop=
9savev2_adam_gru_20_recurrent_kernel_m_read_readvariableop1
-savev2_adam_gru_20_bias_m_read_readvariableop=
9savev2_adam_embedding_20_embeddings_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop3
/savev2_adam_gru_20_kernel_v_read_readvariableop=
9savev2_adam_gru_20_recurrent_kernel_v_read_readvariableop1
-savev2_adam_gru_20_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_d4e99c10432042378b3a14c5d994c7ea/part*
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
: Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Є
SaveV2/tensor_namesConst"/device:CPU:0*░
valueдBБB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Ф
SaveV2/shape_and_slicesConst"/device:CPU:0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_embedding_20_embeddings_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_gru_20_kernel_read_readvariableop2savev2_gru_20_recurrent_kernel_read_readvariableop&savev2_gru_20_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_embedding_20_embeddings_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop/savev2_adam_gru_20_kernel_m_read_readvariableop9savev2_adam_gru_20_recurrent_kernel_m_read_readvariableop-savev2_adam_gru_20_bias_m_read_readvariableop9savev2_adam_embedding_20_embeddings_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop/savev2_adam_gru_20_kernel_v_read_readvariableop9savev2_adam_gru_20_recurrent_kernel_v_read_readvariableop-savev2_adam_gru_20_bias_v_read_readvariableop"/device:CPU:0*-
dtypes#
!2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
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
:ќ
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

identity_1Identity_1:output:0*ј
_input_shapesЧ
щ: :
Њжd:dd:d:d:: : : : : :	dг:	dг:	г: : :
Њжd:dd:d:d::	dг:	dг:	г:
Њжd:dd:d:d::	dг:	dг:	г: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : :  : : : : : :
 : : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : 
я+
├
while_body_825232
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
biasadd_1_unstackѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         dЦ
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гj
BiasAddBiasAddMatMul:product:0biasadd_unstack_0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гp
	BiasAdd_1BiasAddMatMul_1:product:0biasadd_1_unstack_0*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:         dЇ
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
: Ѓ

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: t

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ъ

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ё

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
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*H
_input_shapes7
5: : : : :         d: : ::г::г22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :  : : :	 : :
 
ж>
Х
(__forward_cudnn_gru_with_fallback_826875

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_826737_826876*<
api_implements*(gru_92e3f741-6ed6-4cb8-993a-a8b40f6d89e0*
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
ц0
Ё
*__inference_cudnn_gru_with_fallback_825429

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_592ce9d6-8c49-4d71-8a35-0ba3527a5754*
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
к?
љ
!__inference__wrapped_model_822454
embedding_20_inputL
Hsequential_20_embedding_20_embedding_lookup_read_readvariableop_resource7
3sequential_20_gru_20_statefulpartitionedcall_args_27
3sequential_20_gru_20_statefulpartitionedcall_args_37
3sequential_20_gru_20_statefulpartitionedcall_args_49
5sequential_20_dense_30_matmul_readvariableop_resource:
6sequential_20_dense_30_biasadd_readvariableop_resource9
5sequential_20_dense_31_matmul_readvariableop_resource:
6sequential_20_dense_31_biasadd_readvariableop_resource
identityѕб-sequential_20/dense_30/BiasAdd/ReadVariableOpб,sequential_20/dense_30/MatMul/ReadVariableOpб-sequential_20/dense_31/BiasAdd/ReadVariableOpб,sequential_20/dense_31/MatMul/ReadVariableOpб+sequential_20/embedding_20/embedding_lookupб?sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOpб,sequential_20/gru_20/StatefulPartitionedCall}
sequential_20/embedding_20/CastCastembedding_20_input*

SrcT0*

DstT0*(
_output_shapes
:         жЭ
?sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOpReadVariableOpHsequential_20_embedding_20_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Њжd┤
4sequential_20/embedding_20/embedding_lookup/IdentityIdentityGsequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Њжd╗
+sequential_20/embedding_20/embedding_lookupResourceGatherHsequential_20_embedding_20_embedding_lookup_read_readvariableop_resource#sequential_20/embedding_20/Cast:y:0@^sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*R
_classH
FDloc:@sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         жd▒
6sequential_20/embedding_20/embedding_lookup/Identity_1Identity4sequential_20/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*R
_classH
FDloc:@sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         жd║
6sequential_20/embedding_20/embedding_lookup/Identity_2Identity?sequential_20/embedding_20/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         жdЅ
sequential_20/gru_20/ShapeShape?sequential_20/embedding_20/embedding_lookup/Identity_2:output:0*
T0*
_output_shapes
:r
(sequential_20/gru_20/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:t
*sequential_20/gru_20/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:t
*sequential_20/gru_20/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:║
"sequential_20/gru_20/strided_sliceStridedSlice#sequential_20/gru_20/Shape:output:01sequential_20/gru_20/strided_slice/stack:output:03sequential_20/gru_20/strided_slice/stack_1:output:03sequential_20/gru_20/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: b
 sequential_20/gru_20/zeros/mul/yConst*
value	B :d*
dtype0*
_output_shapes
: ъ
sequential_20/gru_20/zeros/mulMul+sequential_20/gru_20/strided_slice:output:0)sequential_20/gru_20/zeros/mul/y:output:0*
T0*
_output_shapes
: d
!sequential_20/gru_20/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: ў
sequential_20/gru_20/zeros/LessLess"sequential_20/gru_20/zeros/mul:z:0*sequential_20/gru_20/zeros/Less/y:output:0*
T0*
_output_shapes
: e
#sequential_20/gru_20/zeros/packed/1Const*
value	B :d*
dtype0*
_output_shapes
: ▓
!sequential_20/gru_20/zeros/packedPack+sequential_20/gru_20/strided_slice:output:0,sequential_20/gru_20/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:e
 sequential_20/gru_20/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ф
sequential_20/gru_20/zerosFill*sequential_20/gru_20/zeros/packed:output:0)sequential_20/gru_20/zeros/Const:output:0*
T0*'
_output_shapes
:         d╬
,sequential_20/gru_20/StatefulPartitionedCallStatefulPartitionedCall?sequential_20/embedding_20/embedding_lookup/Identity_2:output:0#sequential_20/gru_20/zeros:output:03sequential_20_gru_20_statefulpartitionedcall_args_23sequential_20_gru_20_statefulpartitionedcall_args_33sequential_20_gru_20_statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-822210*(
f#R!
__inference_standard_gru_822209*
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
@:         d:         жd:         d: л
,sequential_20/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_30_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:ddк
sequential_20/dense_30/MatMulMatMul5sequential_20/gru_20/StatefulPartitionedCall:output:04sequential_20/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d╬
-sequential_20/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_30_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:d╗
sequential_20/dense_30/BiasAddBiasAdd'sequential_20/dense_30/MatMul:product:05sequential_20/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d~
sequential_20/dense_30/ReluRelu'sequential_20/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         dл
,sequential_20/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_31_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:d║
sequential_20/dense_31/MatMulMatMul)sequential_20/dense_30/Relu:activations:04sequential_20/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╬
-sequential_20/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_31_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:╗
sequential_20/dense_31/BiasAddBiasAdd'sequential_20/dense_31/MatMul:product:05sequential_20/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
sequential_20/dense_31/SigmoidSigmoid'sequential_20/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:         К
IdentityIdentity"sequential_20/dense_31/Sigmoid:y:0.^sequential_20/dense_30/BiasAdd/ReadVariableOp-^sequential_20/dense_30/MatMul/ReadVariableOp.^sequential_20/dense_31/BiasAdd/ReadVariableOp-^sequential_20/dense_31/MatMul/ReadVariableOp,^sequential_20/embedding_20/embedding_lookup@^sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp-^sequential_20/gru_20/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*G
_input_shapes6
4:         ж::::::::2\
,sequential_20/dense_31/MatMul/ReadVariableOp,sequential_20/dense_31/MatMul/ReadVariableOp2\
,sequential_20/gru_20/StatefulPartitionedCall,sequential_20/gru_20/StatefulPartitionedCall2ѓ
?sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp?sequential_20/embedding_20/embedding_lookup/Read/ReadVariableOp2Z
+sequential_20/embedding_20/embedding_lookup+sequential_20/embedding_20/embedding_lookup2\
,sequential_20/dense_30/MatMul/ReadVariableOp,sequential_20/dense_30/MatMul/ReadVariableOp2^
-sequential_20/dense_31/BiasAdd/ReadVariableOp-sequential_20/dense_31/BiasAdd/ReadVariableOp2^
-sequential_20/dense_30/BiasAdd/ReadVariableOp-sequential_20/dense_30/BiasAdd/ReadVariableOp: : : : : :2 .
,
_user_specified_nameembedding_20_input: : : 
ж>
Х
(__forward_cudnn_gru_with_fallback_825568

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
split_1_split_dimѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0»
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ё
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

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
backward_function_name><__inference___backward_cudnn_gru_with_fallback_825430_825569*<
api_implements*(gru_592ce9d6-8c49-4d71-8a35-0ba3527a5754*
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
оJ
ы
__inference_standard_gru_826647

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_826540*
_num_original_outputs*
bodyR
while_body_826541*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:о
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
:Є
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
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  d«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_92e3f741-6ed6-4cb8-993a-a8b40f6d89e0*
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
щ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_827716

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
:         dг
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-827486*(
f#R!
__inference_standard_gru_827485*
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
@:         d:         жd:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
╩
І
while_cond_826131
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
5: : : : :         d: : ::г::г: : : : : :  : : :	 : :
 
ЬЊ
┘
<__inference___backward_cudnn_gru_with_fallback_823968_824107
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
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
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dѓ
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
:і
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :И┘ќ
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_96187d1d-8adb-4d13-9661-83b3f69b971e*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_824106*╩
_input_shapesИ
х:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ЬЊ
┘
<__inference___backward_cudnn_gru_with_fallback_823548_823687
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
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
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*4
_output_shapes"
 :                  dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:└
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dѓ
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
:і
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*U
_output_shapesC
A:                  d:         d: :И┘ќ
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  dЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_64add08a-754e-4b31-9a7c-25dc3047afac*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_823686*╩
_input_shapesИ
х:         d:                  d:         d: :                  d:::::         d:::: ::                  d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
Ѕ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_823689

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
_gradient_op_typePartitionedCall-823459*(
f#R!
__inference_standard_gru_823458*
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
H:         d:                  d:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*?
_input_shapes.
,:                  d:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
оJ
ы
__inference_standard_gru_823878

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбReadVariableOpбwhileѓ
ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	гa
unstackUnpackReadVariableOp:value:0*	
num*
T0*"
_output_shapes
:г:гc
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
:Л
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
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
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
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         dІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dг|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         гG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Б
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*M
_output_shapes;
9:         d:         d:         dЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	dгn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         гm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         гI
Const_1Const*
value	B :*
dtype0*
_output_shapes
: S
split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Е
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
 *  ђ?*
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
:Б
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
: Щ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*D
output_shapes3
1: : : : :         d: : : :г: :г*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_823771*
_num_original_outputs*
bodyR
while_body_823772*E
_output_shapes3
1: : : : :         d: : : :г: :гK
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
:гM
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: T
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes	
:гЂ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    d   *
dtype0*
_output_shapes
:о
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
:Є
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
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  d[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ф
IdentityIdentitystrided_slice_2:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         d▒

Identity_1Identitytranspose_1:y:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*4
_output_shapes"
 :                  d«

Identity_2Identitywhile/Identity_4:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:         dћ

Identity_3Identityruntime:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_96187d1d-8adb-4d13-9661-83b3f69b971e*
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
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_822299_822438
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_4fe44b6e-07a2-477f-84a1-5ebeb806b94d*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_822437*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ъЊ
┘
<__inference___backward_cudnn_gru_with_fallback_825430_825569
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

identity_4ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         de
gradients/grad_ys_1Identityplaceholder_1*
T0*,
_output_shapes
:         жd`
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
:х
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*,
_output_shapes
:ж         dю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:И
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:ж         dЂ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         dЩ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*,
_output_shapes
:ж         da
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:ѓ
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*
rnn_modegru*M
_output_shapes;
9:ж         d:         d: :И┘ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:Л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         жdu
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
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_2Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_3Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_4Const*
valueB:љN*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_5Const*
valueB:љN*
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
:ў
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::Т
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:љNЖ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:љNж
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:dж
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:dВ
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:do
gradients/Reshape_1_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Б
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_2_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_3_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_4_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_5_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:ddo
gradients/Reshape_6_grad/ShapeConst*
valueB"d   d   *
dtype0*
_output_shapes
:Ц
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:ddh
gradients/Reshape_7_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_8_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:dh
gradients/Reshape_9_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:А
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_10_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:Б
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_11_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:di
gradients/Reshape_12_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:ц
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:dю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:и
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ddј
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:пЁ
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	dгІ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	dгm
gradients/Reshape_grad/ShapeConst*
valueB"   ,  *
dtype0*
_output_shapes
:б
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	гъ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*,
_output_shapes
:         жdЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         dЉ

Identity_2Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгЊ

Identity_3Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	dгћ

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	г"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_592ce9d6-8c49-4d71-8a35-0ba3527a5754*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_gru_with_fallback_825568*▓
_input_shapesа
Ю:         d:         жd:         d: :ж         d:::::         d:::: ::ж         d:         d: :И┘::         d: ::::::: : : 2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :	 : : : : :  : : : : : : : : : : : : : : : : : : :
 
ц0
Ё
*__inference_cudnn_gru_with_fallback_823967

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3ѕбCudnnRNNбReshape/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpG
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
dtype0Г
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
T0*
N=
CudnnRNN/input_cConst*
valueB
 *    *
dtype0ђ
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
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0Ѕ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0ё

Identity_1Identitytranspose_7:y:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ї

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0Ё

Identity_3Identityruntime:output:0	^CudnnRNN^Reshape/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*<
api_implements*(gru_96187d1d-8adb-4d13-9661-83b3f69b971e*
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
▒
н
H__inference_embedding_20_layer_call_and_return_conditional_losses_826054

inputs1
-embedding_lookup_read_readvariableop_resource
identityѕбembedding_lookupб$embedding_lookup/Read/ReadVariableOpV
CastCastinputs*

SrcT0*

DstT0*(
_output_shapes
:         ж┬
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Њжd~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Њжd┤
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceCast:y:0%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*,
_output_shapes
:         жdЯ
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*,
_output_shapes
:         жdё
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         жdФ
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*,
_output_shapes
:         жd"
identityIdentity:output:0*+
_input_shapes
:         ж:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
щ
у
B__inference_gru_20_layer_call_and_return_conditional_losses_827305

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall;
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
:Л
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
B :У*
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
:         dг
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
_gradient_op_typePartitionedCall-827075*(
f#R!
__inference_standard_gru_827074*
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
@:         d:         жd:         d: ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d"
identityIdentity:output:0*7
_input_shapes&
$:         жd:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
R
embedding_20_input<
$serving_default_embedding_20_input:0         ж<
dense_310
StatefulPartitionedCall:0         tensorflow/serving/predict:м╚
П)
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
*k&call_and_return_all_conditional_losses
l__call__
m_default_save_signature"ж&
_tf_keras_sequential╩&{"class_name": "Sequential", "name": "sequential_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_20", "layers": [{"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}, {"class_name": "GRU", "config": {"name": "gru_20", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_20", "layers": [{"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}, {"class_name": "GRU", "config": {"name": "gru_20", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╗
trainable_variables
regularization_losses
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"г
_tf_keras_layerњ{"class_name": "InputLayer", "name": "embedding_20_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 233], "config": {"batch_input_shape": [null, 233], "dtype": "float32", "sparse": false, "name": "embedding_20_input"}}
С

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"┼
_tf_keras_layerФ{"class_name": "Embedding", "name": "embedding_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 233], "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": [null, 233], "dtype": "float32", "input_dim": 29843, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 233}}
ї

cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"с
_tf_keras_layer╔{"class_name": "GRU", "name": "gru_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_20", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 100], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
ш

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
*t&call_and_return_all_conditional_losses
u__call__"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
Ш

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
*v&call_and_return_all_conditional_losses
w__call__"Л
_tf_keras_layerи{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
с
'iter

(beta_1

)beta_2
	*decay
+learning_ratem[m\m]!m^"m_,m`-ma.mbvcvdve!vf"vg,vh-vi.vj"
	optimizer
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
,1
-2
.3
4
5
!6
"7"
trackable_list_wrapper
и
/non_trainable_variables
trainable_variables
regularization_losses
		variables

0layers
1layer_regularization_losses
2metrics
l__call__
m_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
џ
3non_trainable_variables
trainable_variables
regularization_losses
	variables

4layers
5layer_regularization_losses
6metrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
+:)
Њжd2embedding_20/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
џ
7non_trainable_variables
trainable_variables
regularization_losses
	variables

8layers
9layer_regularization_losses
:metrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ь

,kernel
-recurrent_kernel
.bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
*y&call_and_return_all_conditional_losses
z__call__"▓
_tf_keras_layerў{"class_name": "GRUCell", "name": "gru_cell_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell_20", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
џ
?non_trainable_variables
trainable_variables
regularization_losses
	variables

@layers
Alayer_regularization_losses
Bmetrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
!:dd2dense_30/kernel
:d2dense_30/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
џ
Cnon_trainable_variables
trainable_variables
regularization_losses
	variables

Dlayers
Elayer_regularization_losses
Fmetrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_31/kernel
:2dense_31/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
џ
Gnon_trainable_variables
#trainable_variables
$regularization_losses
%	variables

Hlayers
Ilayer_regularization_losses
Jmetrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 :	dг2gru_20/kernel
*:(	dг2gru_20/recurrent_kernel
:	г2gru_20/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
'
K0"
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
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
џ
Lnon_trainable_variables
;trainable_variables
<regularization_losses
=	variables

Mlayers
Nlayer_regularization_losses
Ometrics
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
џ
	Ptotal
	Qcount
R
_fn_kwargs
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
*{&call_and_return_all_conditional_losses
|__call__"т
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
P0
Q1"
trackable_list_wrapper
џ
Wnon_trainable_variables
Strainable_variables
Tregularization_losses
U	variables

Xlayers
Ylayer_regularization_losses
Zmetrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:.
Њжd2Adam/embedding_20/embeddings/m
&:$dd2Adam/dense_30/kernel/m
 :d2Adam/dense_30/bias/m
&:$d2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
%:#	dг2Adam/gru_20/kernel/m
/:-	dг2Adam/gru_20/recurrent_kernel/m
#:!	г2Adam/gru_20/bias/m
0:.
Њжd2Adam/embedding_20/embeddings/v
&:$dd2Adam/dense_30/kernel/v
 :d2Adam/dense_30/bias/v
&:$d2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
%:#	dг2Adam/gru_20/kernel/v
/:-	dг2Adam/gru_20/recurrent_kernel/v
#:!	г2Adam/gru_20/bias/v
Ы2№
I__inference_sequential_20_layer_call_and_return_conditional_losses_826017
I__inference_sequential_20_layer_call_and_return_conditional_losses_825047
I__inference_sequential_20_layer_call_and_return_conditional_losses_825585
I__inference_sequential_20_layer_call_and_return_conditional_losses_825065└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
.__inference_sequential_20_layer_call_fn_825096
.__inference_sequential_20_layer_call_fn_825128
.__inference_sequential_20_layer_call_fn_826043
.__inference_sequential_20_layer_call_fn_826030└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
!__inference__wrapped_model_822454┬
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *2б/
-і*
embedding_20_input         ж
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Ы2№
H__inference_embedding_20_layer_call_and_return_conditional_losses_826054б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_embedding_20_layer_call_fn_826060б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
B__inference_gru_20_layer_call_and_return_conditional_losses_826878
B__inference_gru_20_layer_call_and_return_conditional_losses_826469
B__inference_gru_20_layer_call_and_return_conditional_losses_827305
B__inference_gru_20_layer_call_and_return_conditional_losses_827716Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 2Ч
'__inference_gru_20_layer_call_fn_827724
'__inference_gru_20_layer_call_fn_826886
'__inference_gru_20_layer_call_fn_826894
'__inference_gru_20_layer_call_fn_827732Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_dense_30_layer_call_and_return_conditional_losses_827743б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_30_layer_call_fn_827750б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_31_layer_call_and_return_conditional_losses_827761б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_31_layer_call_fn_827768б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
>B<
$__inference_signature_wrapper_825151embedding_20_input
─2┴Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
─2┴Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 И
I__inference_sequential_20_layer_call_and_return_conditional_losses_826017k,-.!"8б5
.б+
!і
inputs         ж
p 

 
ф "%б"
і
0         
џ ─
I__inference_sequential_20_layer_call_and_return_conditional_losses_825047w,-.!"DбA
:б7
-і*
embedding_20_input         ж
p

 
ф "%б"
і
0         
џ И
I__inference_sequential_20_layer_call_and_return_conditional_losses_825585k,-.!"8б5
.б+
!і
inputs         ж
p

 
ф "%б"
і
0         
џ ─
I__inference_sequential_20_layer_call_and_return_conditional_losses_825065w,-.!"DбA
:б7
-і*
embedding_20_input         ж
p 

 
ф "%б"
і
0         
џ ю
.__inference_sequential_20_layer_call_fn_825096j,-.!"DбA
:б7
-і*
embedding_20_input         ж
p

 
ф "і         ю
.__inference_sequential_20_layer_call_fn_825128j,-.!"DбA
:б7
-і*
embedding_20_input         ж
p 

 
ф "і         љ
.__inference_sequential_20_layer_call_fn_826043^,-.!"8б5
.б+
!і
inputs         ж
p 

 
ф "і         љ
.__inference_sequential_20_layer_call_fn_826030^,-.!"8б5
.б+
!і
inputs         ж
p

 
ф "і         б
!__inference__wrapped_model_822454},-.!"<б9
2б/
-і*
embedding_20_input         ж
ф "3ф0
.
dense_31"і
dense_31         Г
H__inference_embedding_20_layer_call_and_return_conditional_losses_826054a0б-
&б#
!і
inputs         ж
ф "*б'
 і
0         жd
џ Ё
-__inference_embedding_20_layer_call_fn_826060T0б-
&б#
!і
inputs         ж
ф "і         жd├
B__inference_gru_20_layer_call_and_return_conditional_losses_826878},-.OбL
EбB
4џ1
/і,
inputs/0                  d

 
p 

 
ф "%б"
і
0         d
џ ├
B__inference_gru_20_layer_call_and_return_conditional_losses_826469},-.OбL
EбB
4џ1
/і,
inputs/0                  d

 
p

 
ф "%б"
і
0         d
џ ┤
B__inference_gru_20_layer_call_and_return_conditional_losses_827305n,-.@б=
6б3
%і"
inputs         жd

 
p

 
ф "%б"
і
0         d
џ ┤
B__inference_gru_20_layer_call_and_return_conditional_losses_827716n,-.@б=
6б3
%і"
inputs         жd

 
p 

 
ф "%б"
і
0         d
џ ї
'__inference_gru_20_layer_call_fn_827724a,-.@б=
6б3
%і"
inputs         жd

 
p

 
ф "і         dЏ
'__inference_gru_20_layer_call_fn_826886p,-.OбL
EбB
4џ1
/і,
inputs/0                  d

 
p

 
ф "і         dЏ
'__inference_gru_20_layer_call_fn_826894p,-.OбL
EбB
4џ1
/і,
inputs/0                  d

 
p 

 
ф "і         dї
'__inference_gru_20_layer_call_fn_827732a,-.@б=
6б3
%і"
inputs         жd

 
p 

 
ф "і         dц
D__inference_dense_30_layer_call_and_return_conditional_losses_827743\/б,
%б"
 і
inputs         d
ф "%б"
і
0         d
џ |
)__inference_dense_30_layer_call_fn_827750O/б,
%б"
 і
inputs         d
ф "і         dц
D__inference_dense_31_layer_call_and_return_conditional_losses_827761\!"/б,
%б"
 і
inputs         d
ф "%б"
і
0         
џ |
)__inference_dense_31_layer_call_fn_827768O!"/б,
%б"
 і
inputs         d
ф "і         ╝
$__inference_signature_wrapper_825151Њ,-.!"RбO
б 
HфE
C
embedding_20_input-і*
embedding_20_input         ж"3ф0
.
dense_31"і
dense_31         