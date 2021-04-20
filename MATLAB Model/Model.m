syms l1 l2 lp x2 x1 dx1 dx2 x0 y0 dy0 m0 m1 m2 mp tp dtp g t ddx1 ddx2 ddtp
cost1=((l1^2+(x2-x1)^2-l2^2)/(2*(x2-x1)*l1));
sint1=sqrt((1-cost1^2));
x0=x1+l1*cost1;
y0=l1*sint1;
dx0=diff(x0,x1)*dx1+diff(x0,x2)*dx2;
dy0=diff(y0,x1)*dx1+diff(y0,x2)*dx2;

U=-g*(m0*y0+(lp*sin(tp)+y0)*mp);
T=1/2*m1*dx1^2+1/2*m2*dx2^2+1/2*m0*(dx0^2+dy0^2)+1/2*mp*((dx0-lp*sin(tp)*dtp)^2+(dy0+lp*cos(tp)*dtp)^2);

L=T-U;
%RHS
RHS=diff(L,x1);
%LHS
%Take the partial dL/dot(dq)
LHS=diff(L,dx1);
%Make the time derivitive of x1=dx1
LHS=subs(LHS,[x1,x2,tp,dx1, dx2, dtp],[t*dx1,t*dx2,t*tp,t*ddx1,t*ddx2, t*ddtp]);
LHS=diff(LHS,t);
LHS=subs(LHS,[t],[1]);
[A1,b1]=equationsToMatrix(LHS==RHS,[ddx1,ddx2,ddtp]);

%RHS
RHS=diff(L,x2);
%LHS
%Take the partial dL/dot(dq)
LHS=diff(L,dx2);
%Make the time derivitive of x1=dx1
LHS=subs(LHS,[x1,x2,tp,dx1, dx2, dtp],[t*dx1,t*dx2,t*tp,t*ddx1,t*ddx2, t*ddtp]);
LHS=diff(LHS,t);
LHS=subs(LHS,[t],[1]);
[A2,b2]=equationsToMatrix(LHS==RHS,[ddx1,ddx2,ddtp]);

%RHS
RHS=diff(L,tp);
%LHS
%Take the partial dL/dot(dq)
LHS=diff(L,dtp);
%Make the time derivitive of x1=dx1
LHS=subs(LHS,[x1,x2,tp,dx1, dx2, dtp],[t*dx1,t*dx2,t*tp,t*ddx1,t*ddx2, t*ddtp]);
LHS=diff(LHS,t);
LHS=subs(LHS,[t],[1]);
[A3,b3]=equationsToMatrix(LHS==RHS,[ddx1,ddx2,ddtp]);
A=[A1;A2;A3];
b=[b1;b2;b3];

Ap=subs(A,[l1,l2,lp,m1,m2,m0,mp,g],[1,1,1,1,1,1,1,-10]);
bp=subs(b,[l1,l2,lp,m1,m2,m0,mp,g],[1,1,1,1,1,1,1,-10]);

final = inv(Ap)\bp;


