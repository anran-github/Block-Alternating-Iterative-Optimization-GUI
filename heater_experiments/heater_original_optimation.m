function [u_ii, p_ii, theta_ii] = heater_original_optimation(x,r, Ad, Bd,R,Q,K,G)
     
    x_r = r;
    
    % variables to optimize
    u = sdpvar(1,1,'full');
    P=sdpvar(2,2,'symmetric');
    % theta=sdpvar(1,1,'full');
    theta=0.0001;

    
    Objective = R*(norm(u))^2+(Ad*x+Bd*u-x_r)'*Q*(Ad*x+Bd*u-x_r)+((x-x_r)'*P*(x-x_r));
    Constraints = [P>=1e-9;theta>=0;
      % ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))^(0.5)-((x-x_r)'*P*(x-x_r))^(0.5)<=-theta*((x-x_r)'*(x-x_r))^(0.5)];
    ((x-x_r)'*P*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
    ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
    
    % opt=sdpsettings('solver','bmibnb');
    u0=-K*x+G*r(2);
    assign(u,u0);
    assign(P,eye(2));
    opt=sdpsettings('solver','fmincon','MaxIter',2000,'usex0',1);
    sol=optimize(Constraints,0.00000001*Objective,opt)
    u_ii = double(u);
    p_ii = double(P);
    theta_ii=double(theta);

end