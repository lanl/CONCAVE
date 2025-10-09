#-

Symbols M,R,alpha;

Symbols O1,O2;
Functions Op;

Indices i,j,k,l;

Function conj;

Function oprod;

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

#procedure sq(x)
+ conj('x')*('x')
#endprocedure

#procedure sqs(x,c)
+ 'c'*conj('x')*('x')
#endprocedure

#define ci "0"
#procedure sqc(x)
#redefine ci "{'ci'+1}"
#call sqs('x',c('ci'))
#endprocedure

Local hamiltonian = (px^2 + py^2 + pz^2)/(2*M) - alpha*q1 - alpha*q2;
*Local sos =
*#call sq(px-i_*x*q1)
*#call sq(py-i_*y*q1)
*#call sq(pz-i_*z*q1)
*#call sq(px-i_*(x-R)*q2)
*#call sq(py-i_*y*q2)
*#call sq(pz-i_*z*q2)
*;

#if 1
Local sos =
#do coef = {1,i_,-1,-i_}
#do pop = {px,py,pz,px^2,py^2,pz^2}
#do xop = {1,x,y,z,x^2,y^2,z^2}
#do qop = {1,q1,q2,(q1+q2),(q1-q2)}
#call sqc('pop'+'coef'*'xop'*'qop')
#enddo
#enddo
#enddo
#enddo
;
#endif

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
id conj(1) = 1;
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
id px*q2 = q2*px + i_ * (x-R) * q2^3;
id py*q2 = q2*py + i_ * y * q2^3;
id pz*q2 = q2*pz + i_ * z * q2^3;
id q1^2*z^2 = 1 - (x^2 + y^2)*q1^2;
id q2^2*z^2 = 1 - ((x-R)^2 + y^2)*q2^2;
endrepeat;

repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);

*AntiBracket c,i_;

* In order to parse the file, we have to get rid of constants like M and R.
id M = 1;
id M^-1 = 1;
id R = 1;
id alpha = 1;

.sort

#write <hamiltonian.expr> "%E",hamiltonian
#write <sos.expr> "%E",sos
*Print;

.end

