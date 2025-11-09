#define ORDER "20"
#define GENERATORS "px,py,pz,px^2,py^2,pz^2,x,y,z,x*q1,y*q1,z*q1,x*q2,y*q2,z*q2,q1*q2,q1^2,q2^2,px*x,px*y,px*z,py*x,py*y,py*z,pz*x,pz*y,pz*z,px*q1,py*q1,pz*q1,px*q2,py*q2,pz*q2"
#-

#procedure generators(?ops);
#define n "0"
#define expr ""
#do op = {'?ops'}
#define n "{'n'+1}"
#define expr "'expr' + c(i,'n')*'op'"
#enddo
id A(i?) = 'expr';
argument;
id A(i?) = 'expr';
endargument;
#endprocedure

#procedure sos(k)
#define sos ""
#do i = 1,'k'
#define sos "'sos' + conj(A('i'))*A('i')"
#enddo
Local sos = 'sos';
#endprocedure

Symbols M,R,alpha;

Symbols O1,O2;
Functions Op;

Indices i,j,k,l;

Function conj;

Functions oprod,coef;

Tensor c;

Functions px,py,pz;
Functions x,y,z;
Functions q1,q2;
set opset : px,py,pz,x,y,z,q1,q2;

Function A;

CommuteInSet {x,y,z,q1,q2}, {px,py,pz};
CommuteInSet {x,py}, {x,pz};
CommuteInSet {y,px}, {y,pz};
CommuteInSet {z,px}, {z,py};

Local hamiltonian = (px^2 + py^2 + pz^2)/(2*M) - alpha*q1 - alpha*q2;
#call sos('ORDER')
#call generators('GENERATORS')

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
id px*x = x*px - i_;
id py*y = y*py - i_;
id pz*z = z*pz - i_;
id px*q1 = q1*px + i_ * x * q1^3;
id py*q1 = q1*py + i_ * y * q1^3;
id pz*q1 = q1*pz + i_ * z * q1^3;
* TODO check these
id px*q2 = q2*px + i_ * (x-R) * q2^3;
id py*q2 = q2*py + i_ * y * q2^3;
id pz*q2 = q2*pz + i_ * z * q2^3;
id q1^2*z^2 = 1 - (x^2 + y^2)*q1^2;
id q2^2*z^2 = 1 - ((x-R)^2 + y^2)*q2^2;
endrepeat;

* Collect into dummy functions
repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);
id conj(c(i?,j?))*c(k?,l?) = coef(i,j,k,l);

Print +s;

.end

