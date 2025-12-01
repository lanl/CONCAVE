#-

#procedure sos(?ops)
#define sos ""
#define i "0"
#define j "0"
#do opi = {'?ops'}
#redefine j "0"
#do opj = {'?ops'}
#redefine sos "'sos' + M('i','j') * conj('opi') * 'opj'"
* TODO do the ground-state constraints too
#redefine j "{'j'+1}"
#enddo
#redefine i "{'i'+1}"
#enddo
Local sos = 'sos';
#endprocedure

Symbols O1,O2;
Functions Op;
Indices i,j,k,l;
Functions conj, oprod;

Commuting M;
Commuting G;

set opset : ;

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

* Perform commutations.
repeat;
endrepeat;

Print +s;

.end

