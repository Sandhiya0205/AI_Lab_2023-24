# Ex.No: 7  Logic Programming –  Logic Circuit Design
### DATE:    9/9/25                                                                        
### REGISTER NUMBER : 212223060239
### AIM: 
To write a logic program to design a circuit like half adder and half subtractor.
###  Algorithm:
1. Start the Program
2. Design a AND gate logic if both inputs are 1 then output is 1.
3. Design a OR gate logic if any one of input is 1 then output is 1.
4. Design a XOR gate logic if both inputs are different then output is 1.
5. Design a NOT gate logic if input is 0 then output is 1.
6. Design a half adder and half subtractor using the rules.
7. Test the logic.
8. Stop the program.

### Program:
```
xor(0,1,1).
xor(0,0,0).
xor(1,0,1).
xor(1,1,0).
and(1,1,1).
and(0,0,0).
and(0,1,0).
and(1,0,0).
not(0,1).
not(1,0).
or(0,1,1).
or(1,0,1).
or(0,0,0).
or(1,1,1).
halfadder(A,B,Sum,Carry):-
    xor(A,B,Sum),
    and(A,B,Carry).
halfsubtractor(A,B,Diff,Carry):-
    xor(A,B,Diff),
    not(A,C),
     and(C,B,Carry).
fulladder(A,B,Cin,S,Cout):-
    xor(A,B,X),
    xor(X,Cin,S),
    and(X,Cin,Y),
    and(A,B,Z),
    or(Y,Z,Cout).
```

### Output:
<img width="466" height="274" alt="image" src="https://github.com/user-attachments/assets/f75dd297-71cd-4a7c-b772-e2a1f5908570" />

<img width="515" height="193" alt="image" src="https://github.com/user-attachments/assets/a523b9b6-ee12-4cce-8662-671d3067a9a5" />

<img width="508" height="189" alt="image" src="https://github.com/user-attachments/assets/dc95ccc2-ebb2-41b0-b3a2-ae2da2bfb308" />




### Result:
Thus the truth table of circuit verified sucessfully.
