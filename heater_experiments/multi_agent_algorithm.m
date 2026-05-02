function [ui,pit,thetai] = multi_agent_algorithm(x,x_r,Ad, Bd,R,Q,K,G)

    % start with original YALMIP method.
    [u_init,p_init,theta_init] = heater_original_optimation(x,x_r,Ad, Bd,R,Q,K,G);
    
    % init errors, start while loop until stop threshold is satisfied.
    delta = 0.01;
    error1 = 1;
    error2 = 1;
    error3 = 1;
    error_cost = 1;
    iteration = 1;
    
    while iteration <= 1e3
        
        % if error1 <= delta && error2 <= delta && error3 <= delta
        if error_cost <= delta
            break
        end
        % start with optimize u      
        if iteration == 1
            pit = p_init;
            ui = u_init;
            thetai=theta_init;
    
        end
        % ==================OPT U=======================
        clear yalmip
        u = sdpvar(1,1,'full');
        % theta=sdpvar(1,1,'full');
        theta=0.0001;
        % slack = sdpvar(1,1,'full');
    
        Objective1 = R*(norm(u))^2+ (Ad*x+Bd*u-x_r)'*Q*(Ad*x+Bd*u-x_r)+((x-x_r)'*pit*(x-x_r))+exp(-theta);
        Constraints = [
            ((Ad*x+Bd*u-x_r)'*pit*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
            % ((Ad*x+Bd*u-x_r)'*pit*(Ad*x+Bd*u-x_r))^(0.5)-((x-x_r)'*pit*(x-x_r))^(0.5)<=-theta*((x-x_r)'*(x-x_r))^(0.5)];
           
    
        opt=sdpsettings('solver','fmincon');
        sol=optimize(Constraints,0.0001*Objective1,opt)
    
        % cost1(iteration) = double(Objective1);
        % update u with optimized solution
        u_ii = double(u);
        theta_ii = double(theta);
    
        % ==================OPT P=======================
        clear yalmip
        P=sdpvar(2,2,'symmetric');
        Slack=sdpvar(1,1,'full');
        
        Objective2 = R*(norm(ui))^2+ (Ad*x+Bd*ui-x_r)'*Q*(Ad*x+Bd*ui-x_r)+((x-x_r)'*P*(x-x_r))+exp(-thetai);
        Constraints = [P>=1e-9;
            % ((Ad*x+Bd*ui-x_r)'*P*(Ad*x+Bd*ui-x_r))^(0.5)-((x-x_r)'*P*(x-x_r))^(0.5)<=-thetai*((x-x_r)'*(x-x_r))^(0.5)];
        ((x-x_r)'*P*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
        ((Ad*x+Bd*ui-x_r)'*P*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
    
    
        sol=optimize(Constraints,0.001*Objective2)
        % cost2(iteration) = double(Objective2);
        
        % update p with optimal solution
        p_ii = double(P);
    
       
        % update pi and ui with w1 and w2
        % w1=0.9;
        % w2=0.1;
        % ui = w1*u_ii+w2*ui;
        % pit = w1*p_ii+w2*pit;
        % thetai=w1*theta_ii+w2*thetai;
        % ================== update pi, ui, thetai with optimal w ===========
        w1 = sdpvar(1,1,'full');
        w2 = sdpvar(1,1,'full');
        w3 = sdpvar(1,1,'full');
    
        Slack=sdpvar(1,1,'full');
        ui = w1*u_ii+(1-w1)*ui;
        pit = w2*p_ii+(1-w2)*pit;
        thetai=w3*theta_ii+(1-w3)*thetai;
        Objective3 = 0.1*(norm(ui))^2+ 2*(norm(Ad*x+Bd*(ui)-x_r)^2)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
        Constraints = [pit>=Slack;Slack>=1e-10;
            w1>=0;w1<=1;w2>=0;w2<=1;w3>=0;w3<=1;
        ((x-x_r)'*pit*(x-x_r))>=((1.5*thetai)^2)*(x-x_r)'*(x-x_r);
        ((Ad*x+Bd*ui-x_r)'*pit*(Ad*x+Bd*ui-x_r))<=((0.5*thetai)^2)*(x-x_r)'*(x-x_r)];
        sol=optimize(Constraints,Objective3,opt)
    
        % double([w1,w2,w3])
        ui = double(ui);
        pit = double(pit);
        thetai = double(thetai);


         % update errors:
        error1 = norm(u_ii-ui);
        error2 = norm(p_ii-pit);
        error3 = norm(theta_ii-thetai);
    
        Cost(iteration)=R*(norm(ui))^2+ (Ad*x+Bd*ui-x_r)'*Q*(Ad*x+Bd*ui-x_r)+((x-x_r)'*pit*(x-x_r))+exp(-thetai);
        if iteration > 1
            error_cost = abs(Cost(iteration) - Cost(iteration-1));
        else
            error_cost = abs(Cost(iteration));
        end
        [Cost(iteration),error_cost,error1,error2,error3]


        iteration = iteration + 1;   
    end
    % plot(Cost)
end

