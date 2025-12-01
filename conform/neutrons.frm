#-

* TODO impose equations of motion, somehow

#procedure sos(?ops)
#define sos ""
#define i "0"
#define j "0"
#do opi = {'?ops'}
#redefine j "0"
#do opj = {'?ops'}
#redefine sos "'sos' + M('i','j') * conj('opi') * 'opj'"
#redefine sos "'sos' + G('i','j') * conj('opi') * (hamiltonian*'opj' - 'opj'*hamiltonian)"
#redefine j "{'j'+1}"
#enddo
#redefine i "{'i'+1}"
#enddo
Local sos = 'sos';
#endprocedure

#procedure hamiltonian(L)
#define ham ""
#do x = 1,'L'
#do y = 1,'L'
#do z = 1,'L'
#redefine ham "'ham' + {'x'*'y'}"
#enddo
#enddo
#enddo
Local hamiltonian = 'ham';
#endprocedure

Symbols O1,O2;
Functions Op;
Indices i,j,k,l;
Functions conj, oprod;

Function psi,psidag;
set opset : psi,psidag;

Commuting M;
Commuting G;

#call hamiltonian(3)
*#call sos('GENERATORS')

* Multiple arguments to conj(): addition.
SplitArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O1)+conj(O2,?a);
Normalize conj;

* Multiple arguments to conj(): multiplication.
FactArg conj;
Repeat id conj(O1?,O2?,?a) = conj(O2,?a)*conj(O1);
id conj(i_) = -i_;

* Collect into dummy functions
repeat id Op?opset = oprod(Op);
repeat id oprod(?O1)*oprod(?O2) = oprod(?O1,?O2);

* Conjugate operators
id conj(psi) = psidag;
id conj(psidag) = psi;

* Perform commutations.
repeat;
* Trivial commutations

* Nontrivial commutations
endrepeat;

Print +s;

.end

