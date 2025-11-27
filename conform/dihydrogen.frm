#define GENERATORS "px,py,pz,x*q1,y*q1,z*q1,x*q2,y*q2,z*q2,q2"
#redefine GENERATORS "'GENERATORS',px*px,px*py,px*pz,py*py,py*pz,pz*pz"
#redefine GENERATORS "'GENERATORS',x*x*q1^2,x*y*q1^2,x*z*q1^2,y*y*q1^2,y*z*q1^2,z*z*q1^2"
#redefine GENERATORS "'GENERATORS',x*x*q2^2,x*y*q2^2,x*z*q2^2,y*y*q2^2,y*z*q2^2,z*z*q2^2"
#redefine GENERATORS "'GENERATORS',x*x*q1*q2,x*y*q1*q2,x*z*q1*q2,y*y*q1*q2,y*z*q1*q2,z*z*q1*q2"
#redefine GENERATORS "'GENERATORS',x*q1*q2,y*q1*q2,z*q1*q2,x*q2^2,y*q2^2,z*q2^2"

*#define GENERATORS "px,py,pz,x*q1,y*q1,z*q1,x*q2,y*q2,z*q2,q1,q2"
*#redefine GENERATORS "'GENERATORS',x,y,z,x^2,y^2,z^2,x*y,x*z,y*z"
*#redefine GENERATORS "'GENERATORS',x^2*q1,x^2*q2,y^2*q1,y^2*q2,z^2*q1,z^2*q2"
*#redefine GENERATORS "'GENERATORS',x^3*q1,x^3*q2,y^3*q1,y^3*q2,z^3*q1,z^3*q2,q1*q2,q1^2,q2^2"
*#redefine GENERATORS "'GENERATORS',px*x,py*y,pz*z,px*y,px*z,py*x,py*z,pz*x,pz*y"
*#redefine GENERATORS "'GENERATORS',px*q1,py*q1,pz*q1,px*q2,py*q2,pz*q2"
*#redefine GENERATORS "'GENERATORS',x*y,x*z,y*z"
*#redefine GENERATORS "'GENERATORS',px^2,py^2,pz^2,px*py,px*pz,py*pz"
*#redefine GENERATORS "'GENERATORS',q1^3,q1^2*q2,q1*q2^2,q2^3"
#-

#procedure sos(?ops)
#define sos ""
#define i "0"
#define j "0"
#do opi = {'?ops'}
#redefine j "0"
#do opj = {'?ops'}
#redefine sos "'sos' + M('i','j') * conj('opi') * 'opj'"
#redefine j "{'j'+1}"
#enddo
#redefine i "{'i'+1}"
#enddo
Local sos = 'sos';
#endprocedure

Symbols m,minv,R,R2,alpha;
Symbols O1,O2;
Functions Op;

Indices i,j,k,l;

Function conj;

Functions oprod;

Tensor c;

Functions px,py,pz;
Functions x,y,z;
Functions q1,q2;
set opset : px,py,pz,x,y,z,q1,q2;

Function M;

Local hamiltonian = (px^2 + py^2 + pz^2)/(2*m) - alpha*q1 - alpha*q2;
#call sos('GENERATORS')

* Multiple arguments to conj(): addition.
SplitArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O1)+conj(O2,?a);
Normalize conj;

* Multiple arguments to conj(): multiplication.
FactArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O2,?a)*conj(O1);
id conj(i_) = -i_;
id conj(R) = R;

* Conjugate operators
id conj(x) = x;
id conj(y) = y;
id conj(z) = z;
id conj(q1) = q1;
id conj(q2) = q2;
id conj(px) = px;
id conj(py) = py;
id conj(pz) = pz;

* Perform commutations.
repeat;
* Trivial
id y*x = x*y;
id z*x = x*z;
id z*y = y*z;
id py*px = px*py;
id pz*px = px*pz;
id pz*py = py*pz;
id px*y = y*px;
id px*z = z*px;
id py*x = x*py;
id py*z = z*py;
id pz*x = x*pz;
id pz*y = y*pz;
id q1*x = x*q1;
id q1*y = y*q1;
id q1*z = z*q1;
id q2*x = x*q2;
id q2*y = y*q2;
id q2*z = z*q2;
id q2*q1 = q1*q2;

* Non-trivial
id px*x = x*px - i_;
id py*y = y*py - i_;
id pz*z = z*pz - i_;
id px*q1 = q1*px + i_ * x * q1^3;
id py*q1 = q1*py + i_ * y * q1^3;
id pz*q1 = q1*pz + i_ * z * q1^3;
id px*q2 = q2*px + i_ * (x-R) * q2^3;
id py*q2 = q2*py + i_ * y * q2^3;
id pz*q2 = q2*pz + i_ * z * q2^3;
id z^2*q1^2 = 1 - (x^2 + y^2)*q1^2;
id z^2*q2^2 = 1 - ((x-R)^2 + y^2)*q2^2;
id z^2*q1*q2^2 = (1 - ((x-R)^2 + y^2)*q2^2)*q1;
endrepeat;

* Collect into dummy functions
repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);

id m^-1 = minv;

Print +s;

.end

