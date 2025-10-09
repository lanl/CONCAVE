#define ORDER "3"
#define GENERATORS "px,py,pz,x*q,y*q,z*q"
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
Functions q;
set opset : px,py,pz,x,y,z,q;

Function A;

CommuteInSet {x,y,z,q}, {px,py,pz};
CommuteInSet {x,py}, {x,pz};
CommuteInSet {y,px}, {y,pz};
CommuteInSet {z,px}, {z,py};

* Hamiltonian and SOS ansatz
Local hamiltonian = (px^2 + py^2 + pz^2)/(2*M) - alpha*q;
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
id px*q = q*px + i_ * x * q^3;
id py*q = q*py + i_ * y * q^3;
id pz*q = q*pz + i_ * z * q^3;
id px*x = x*px - i_;
id py*y = y*py - i_;
id pz*z = z*pz - i_;
*id px*y = y*px;
*id px*z = z*px;
*id py*x = x*py;
*id py*z = z*py;
*id pz*x = x*pz;
*id pz*y = y*pz;
id q^2*z^2 = 1 - (x^2 + y^2)*q^2;
endrepeat;

* Collect into dummy functions
repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);
*id conj(c(i?,j?)) * oprod(?O1)*c(k?,l?) = coef(i,j,k,l) * oprod(?O1);

*id c(1,1) = 1;
*id conj(c(1,1)) = 1;
*id c(2,2) = 1;
*id conj(c(2,2)) = 1;
*id c(3,3) = 1;
*id conj(c(3,3)) = 1;
*id c(1,4) = - alpha * i_;
*id conj(c(1,4)) = alpha * i_;
*id c(2,5) = - alpha * i_;
*id conj(c(2,5)) = alpha * i_;
*id c(3,6) = - alpha * i_;
*id conj(c(3,6)) = alpha * i_;
*
*id c(i?,j?) = 0;
*id conj(c(i?,j?)) = 0;

Print +s;

.end

