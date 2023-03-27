function [Q_d_i,Q_d_m,P_bar_i,DM_i,P_CF_i,I_i,P_bar_h,C_h,P_bar_CF_h,I_h,P_j,C_j,P_l,C_l]=search_and_matching(P_i,Y_i,S_i,S_i_,G_i,P_m,Y_m,a_sg,DM_d_i,b_CF_g,I_d_i,b_HH_g,C_d_h,b_CFH_g,I_d_h,c_G_g,C_d_j,c_E_g,C_d_l)
% Returns all quantities and prices for goods that are traded on markets

% Initialize aggregate numbers of agents
% Number of Goods
G=size(b_HH_g,1);
% Firms
I=size(P_i,2);
% Number of Households
H=size(C_d_h,2);
% Number of Foreign consumers (those that purchase exports)
L=size(C_d_l,2);
% Number of government entities consuming
J=size(C_d_j,2);

% For convenience, domestic and foreign goods are not differentiated
P_f=[P_i,P_m];
S_f=[Y_i+S_i,Y_m];
S_f_=[S_i_,Inf(size(Y_m))];
G_f=[G_i,1:G];

% Supply of products for final uses and intermediate input
% Demand for all domestic firm production initialized
Q_d_i=zeros(size(Y_i));
% Demand for imported goods initialized
Q_d_m=zeros(size(Y_m));

% Investments and intermediate inputs initalized
% Investments and price investments of firm i initalized
I_i=zeros(1,I);
P_CF_i=zeros(1,I);
% Intermediate input and price intermediate inputs of firm i initalized
DM_i=zeros(1,I);
P_bar_i=zeros(1,I);

% Initialize Consumption and investment by households
C_h=zeros(size(C_d_h));
I_h=zeros(size(I_d_h));
C_j=0;
C_l=0;

% Price indices are initialized as scalars
% Households
P_bar_h=0;
P_bar_CF_h=0;
% Gov. cons. agents
P_j=0;
% Foreign consumers
P_l=0;

% Main search and matching loop, runs over product G, where each product is a separate market
% You can let it run in parallel in a parfor loop
% parfor g=1:G
for g=1:G
	% Determine demand for intermediate inputs and investments from aggregate demand variables "DM_d_i" "I_d_i" on firm levels, and using technical coefficients in IOTs "a_sg(g,G_i)" and "b_CF_g(g)"
    % Real demand of firm i for intermediary consumption and investments
    DM_d_ig=a_sg(g,G_i).*DM_d_i+b_CF_g(g)*I_d_i;
	% Initialized as nominal variable
    DM_nominal_ig=zeros(size(DM_d_ig));
	% Find firms with demand according to the market g we loop over
    I_g=find(DM_d_ig>0);
    % All firms that offer this good are found
    F_g=find(G_f==g);
    S_fg=S_f;
    F_g(S_fg(F_g)<=0)=[];
    S_fg_=S_f_;
	% This is the actual search and matching, see below equ. (A.1) in the appendix, p. 1 
    while ~isempty(I_g) && ~isempty(F_g)   % Loop runs until there is either no demand or no supply of a goods
        pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
        pr_size_f=S_f(F_g)/sum(S_f(F_g));
        pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
        I_g=I_g(randperm(length(I_g)));
        for j=1:length(I_g)
            i=I_g(j);
            % Draw a random number from an empirical distribution
            e=randf(pr_cum_f);
            f=F_g(e);
            % A firm satisfies its demand, if it has the item on stock for
            % full amount
            if S_fg(f)>DM_d_ig(i)
                S_fg(f)=S_fg(f)-DM_d_ig(i);
                DM_nominal_ig(i)=DM_nominal_ig(i)+DM_d_ig(i)*P_f(f);
                DM_d_ig(i)=0;
            % if not all demand can be satisfied from stock
            else
                DM_d_ig(i)=DM_d_ig(i)-S_fg(f);
                DM_nominal_ig(i)=DM_nominal_ig(i)+S_fg(f)*P_f(f);
                S_fg(f)=0;
                F_g(e)=[];
                if isempty(F_g)
                    break
                end
                pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
                pr_size_f=S_f(F_g)/sum(S_f(F_g));
                pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
            end
        end
        I_g=find(DM_d_ig>0);
    end
    if ~isempty(I_g)
        DM_d_ig_=DM_d_ig;
        I_g=find(DM_d_ig_>0);
        F_g=find(G_f==g);
        F_g(S_fg_(F_g)<=0|S_f(F_g)<=0)=[];
        while ~isempty(I_g) && ~isempty(F_g)
            pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
            pr_size_f=S_f(F_g)/sum(S_f(F_g));
            pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
            I_g=I_g(randperm(length(I_g)));
            for j=1:length(I_g)
                i=I_g(j);
                e=randf(pr_cum_f);
                f=F_g(e);
                if S_fg_(f)>DM_d_ig_(i)
                    S_fg(f)=S_fg(f)-DM_d_ig_(i);
                    S_fg_(f)=S_fg_(f)-DM_d_ig_(i);
                    DM_d_ig_(i)=0;
                else
                    DM_d_ig_(i)=DM_d_ig_(i)-S_fg_(f);
                    S_fg(f)=S_fg(f)-S_fg_(f);
                    S_fg_(f)=0;
                    F_g(e)=[];
                    if isempty(F_g)
                        break
                    end
                    pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
                    pr_size_f=S_f(F_g)/sum(S_f(F_g));
                    pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
                end
            end
            I_g=find(DM_d_ig_>0);
        end
    end
    
    % Here the temporary variable are collected into permanent ones (this
    % is because of the parallel computing process)
    DM_i=a_sg(g,G_i).*DM_d_i-max(0,DM_d_ig-b_CF_g(g)*I_d_i)+DM_i;
    I_i=I_i+max(0,b_CF_g(g)*I_d_i-DM_d_ig);
    P_bar_i=P_bar_i+max(0,DM_nominal_ig.*(a_sg(g,G_i).*DM_d_i-max(0,DM_d_ig-b_CF_g(g)*I_d_i))./(a_sg(g,G_i).*DM_d_i+b_CF_g(g)*I_d_i-DM_d_ig));
    P_CF_i=P_CF_i+max(0,DM_nominal_ig.*max(0,b_CF_g(g)*I_d_i-DM_d_ig)./(a_sg(g,G_i).*DM_d_i+b_CF_g(g)*I_d_i-DM_d_ig));
    
    % Here the market for household consumption is run
    % This market is in nominal terms (before for intermediary inputs and
    % investment, it was in real terms and then put into nominal values)
    C_d_hg=[b_HH_g(g)*C_d_h+b_CFH_g(g)*I_d_h,c_E_g(g)*C_d_l,c_G_g(g)*C_d_j];
    C_real_hg=zeros(size(C_d_hg));
    H_g=find(C_d_hg>0);
    F_g(S_fg(F_g)<=0)=[];
    while ~isempty(H_g) && ~isempty(F_g)
        pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
        pr_size_f=S_f(F_g)/sum(S_f(F_g));
        pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
        H_g=H_g(randperm(length(H_g)));
        for j=1:length(H_g)
            h=H_g(j);
            e=randf(pr_cum_f);
            f=F_g(e);
            if S_fg(f)>C_d_hg(h)/P_f(f)
                S_fg(f)=S_fg(f)-C_d_hg(h)/P_f(f);
                C_real_hg(h)=C_real_hg(h)+C_d_hg(h)/P_f(f);
                C_d_hg(h)=0;
            else
                C_d_hg(h)=C_d_hg(h)-S_fg(f)*P_f(f);
                C_real_hg(h)=C_real_hg(h)+S_fg(f);
                S_fg(f)=0;
                F_g(e)=[];
                if isempty(F_g)
                    break
                end
                pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
                pr_size_f=S_f(F_g)/sum(S_f(F_g));
                pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
            end
        end
        H_g=find(C_d_hg>0);
    end
    if ~isempty(H_g)
        C_d_hg_=C_d_hg;
        H_g=find(C_d_hg_>0);
        F_g=find(G_f==g);
        F_g(S_fg_(F_g)<=0|S_f(F_g)<=0)=[];
        while ~isempty(H_g) && ~isempty(F_g)
            pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
            pr_size_f=S_f(F_g)/sum(S_f(F_g));
            pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
            H_g=H_g(randperm(length(H_g)));
            for j=1:length(H_g)
                h=H_g(j);
                e=randf(pr_cum_f);
                f=F_g(e);
                if S_fg_(f)>C_d_hg_(h)/P_f(f)
                    S_fg(f)=S_fg(f)-C_d_hg_(h)/P_f(f);
                    S_fg_(f)=S_fg_(f)-C_d_hg_(h)/P_f(f);
                    C_d_hg_(h)=0;
                else
                    C_d_hg_(h)=C_d_hg_(h)-S_fg_(f)*P_f(f);
                    S_fg(f)=S_fg(f)-S_fg_(f);
                    S_fg_(f)=0;
                    F_g(e)=[];
                    if isempty(F_g)
                        break
                    end
                    pr_price_f=max(0,exp(-2*P_f(F_g))./sum(exp(-2*P_f(F_g))));
                    pr_size_f=S_f(F_g)/sum(S_f(F_g));
                    pr_cum_f=[0,cumsum(pr_price_f+pr_size_f)/sum(pr_price_f+pr_size_f)];
                end
            end
            H_g=find(C_d_hg_>0);
        end
    end
    
    Q_d_i=S_f(1:I)-S_fg(1:I)+Q_d_i;
    Q_d_m=S_f(I+1:end)-S_fg(I+1:end)+Q_d_m;
    
    C_h=b_HH_g(g)*C_d_h-max(0,C_d_hg(1:H)-b_CFH_g(g)*I_d_h)+C_h;
    I_h=max(0,b_CFH_g(g)*I_d_h-C_d_hg(1:H))+I_h;
    C_j=sum(c_G_g(g)*C_d_j)-sum(C_d_hg(H+L+1:H+L+J))+C_j;
    C_l=sum(c_E_g(g)*C_d_l)-sum(C_d_hg(H+1:H+L))+C_l;
    
    P_bar_h=P_bar_h+max(0,sum(C_real_hg(1:H))*sum(C_d_h*b_HH_g(g)-max(0,C_d_hg(1:H)-b_CFH_g(g)*I_d_h))/sum((C_d_h*b_HH_g(g)+b_CFH_g(g)*I_d_h-C_d_hg(1:H))));
    P_bar_CF_h=P_bar_CF_h+max(0,sum(C_real_hg(1:H))*sum(max(0,b_CFH_g(g)*I_d_h-C_d_hg(1:H)))/sum((C_d_h*b_HH_g(g)+b_CFH_g(g)*I_d_h-C_d_hg(1:H))));
    P_j=P_j+sum(C_real_hg(H+L+1:H+L+J));
    P_l=P_l+sum(C_real_hg(H+1:H+L));
end

P_CF_i(I_i>0)=P_CF_i(I_i>0)./I_i(I_i>0);
P_bar_i(DM_i>0)=P_bar_i(DM_i>0)./DM_i(DM_i>0);

P_bar_h=sum(C_h)/P_bar_h;
P_bar_CF_h=sum(I_h)/P_bar_CF_h;
P_j=C_j/P_j;
P_l=C_l/P_l;

end

