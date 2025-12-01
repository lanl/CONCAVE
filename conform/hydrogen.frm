#define GENERATORS "px,py,pz,x*q,y*q,z*q"
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

Symbols m,minv,alpha;
Symbols O1,O2;
Functions Op;

Indices i,j,k,l;

Function conj;

Functions oprod;

Functions px,py,pz;
Functions x,y,z;
Functions q;
set opset : px,py,pz,x,y,z,q;

Function M;

* Hamiltonian and SOS ansatz
Local hamiltonian = (px^2 + py^2 + pz^2)/(2*m) - alpha*q;
#call sos('GENERATORS')

* Multiple arguments to conj(): addition.
SplitArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O1)+conj(O2,?a);
Normalize conj;

* Multiple arguments to conj(): multiplication.
FactArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O2,?a)*conj(O1);
id conj(i_) = -i_;

* Conjugate operators
id conj(x) = x;
id conj(y) = y;
id conj(z) = z;
id conj(q) = q;
id conj(px) = px;
id conj(py) = py;
id conj(pz) = pz;

* Perform commutations.
repeat;
* Trivial commutations
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
id q*x = x*q;
id q*y = y*q;
id q*z = z*q;

* Nontrivial commutations
id px*q = q*px + i_ * x * q^3;
id py*q = q*py + i_ * y * q^3;
id pz*q = q*pz + i_ * z * q^3;
id px*x = x*px - i_;
id py*y = y*py - i_;
id pz*z = z*pz - i_;
id z^2*q^2 = 1 - (x^2 + y^2)*q^2;
endrepeat;

* Collect into dummy functions
repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);

id m^-1 = minv;

Print +s;

.end

