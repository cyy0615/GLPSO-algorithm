%拉丁超立方体+反向学习+随机分组学习+线性减小学习因子+最差组速度高斯扰动+std Mfitness 高斯扰动+双面镜  终版GLPSO
function [gbest, gbestval, convergence, fitcount] = GLPSO_func(fhd, Dimension, Particle_Number, Max_Gen, VRmin, VRmax, varargin)
    rand('state', sum(100 * clock)); % 设置随机数种子
    me = Max_Gen; % 最大迭代次数
    ps = Particle_Number; % 粒子数量
    D = Dimension; % 维度
   % cc = [1.5, 1, 1];  % 学习因子  个体 群体 小组
    iwt = 0.9 - (1:me) .* (0.5 ./ me); % 惯性权重
    gc = golden_sequence(0.1, 1.6, me);  % 随机序列
    if length(VRmin) == 1
        VRmin = repmat(VRmin, 1, D);
        VRmax = repmat(VRmax, 1, D);
    end
    cc1 = 2 - (1:me) .* (1.9 ./ me);
    cc2 = 0.6 + (1:me) .* (1 ./ me);
    cc3 = 0.3 + (1:me) .* (1.4 ./ me);

    mv = 0.5 * (VRmax - VRmin); % 移动速度
    VRmin = repmat(VRmin, ps, 1);
    VRmax = repmat(VRmax, ps, 1);
    Vmin = repmat(-mv, ps, 1); % 最小速度
    Vmax = -Vmin; % 最大速度

       % 初始化种群位置
    [lhsPos, oblPos] = initialize_population_with_LHS_OBL(VRmin, VRmax, Particle_Number, Dimension)
    
    % 计算初始适应度
    lhse = feval(fhd, lhsPos', varargin{:});
    oble= feval(fhd, oblPos', varargin{:});
% 假设lhse和oble都是列向量，且长度相同
% 初始化e和pos，长度与lhse和oble相同
  
  pbest = zeros(ps, D); % 初始化为零，之后更新为实际位置


% 初始化e和pbest
for i = 1:ps
    if lhse(i) < oble(i)
        e(i) = lhse(i);
        pos(i,:) = lhsPos(i,:); % 存储lhsPos中的位置
    else
        e(i) = oble(i);
        pos(i,:) = oblPos(i,:); % 存储oblPos中的位置
    end
end

% 初始化全局最优解
   [gbestval, gbestid] = min(e); % 找到全局最优适应度和索引

    fitcount = ps; % 适应度计数
    vel = Vmin + 2 .* Vmax .* rand(ps, D);  % 速度
    pbest = pos; % 个体最优位置
    pbestval = e; % 个体最优适应度
    [gbestval, gbestid] = min(pbestval); % 全局最优适应度和位置
    gbest = pbest(gbestid,:); 
    gbestrep = repmat(gbest, ps, 1);

    convergence = ones(1, me); % 收敛曲线
    convergence(1) = gbestval;


       
    for i = 2:me/2
        % 每个小组的粒子数量
        group_size = 3; 
        % 更新小组最优
        [pbest_group, num_groups_actual] = update_group_best(pbest, pbestval, group_size);
        % 计算每个小组的平均适应度
        group_avg_fitness = zeros(num_groups_actual, 1);
        for j = 1:ps/group_size
            group_indices = (j-1)*group_size+1:j*group_size;
            group_avg_fitness(j) = mean(pbestval(group_indices));
        end
        
        % 找到最差组
        [~, worst_group_idx] = max(group_avg_fitness);
        worst_group_indices = (worst_group_idx-1)*group_size+1:worst_group_idx*group_size;
        
        % 对最差组中的每个粒子进行高斯扰动
        for j = worst_group_indices
       
       % 添加高斯扰动
        perturbation_rate = 0.3; % 扰动概率
        gaussian_perturbation = rand(ps, D) < perturbation_rate;
        gaussian_sigma = 0.1; % 高斯扰动的标准差
        gaussian_vel = gaussian_sigma * randn(ps, D); % 高斯扰动
        % 根据扰动概率随机决定是否使用高斯扰动
        vel = gaussian_vel .* gaussian_perturbation + vel .* ~gaussian_perturbation;
       
        end
        % 更新每个粒子的速度
        aa = cc1(i) .* rand(ps, D) .* (pbest - pos) + cc2(i) .* rand(ps, D) .* (gbestrep - pos)+ cc3(i) .* rand(ps, D) .* (pbest_group - pos);
        vel = iwt(i) .* vel .* gc(i) + aa; % 更新速度
        vel = (vel > Vmax) .* Vmax + (vel <= Vmax) .* vel; % 限制速度范围
        vel = (vel < Vmin) .* Vmin + (vel >= Vmin) .* vel;
      
        pos = pos + vel; % 更新位置
         % 更新粒子位置后，检查边界
        pos = pos .* ((pos >= VRmin) & (pos <= VRmax)) + (pos < VRmin) .* (VRmin + (VRmax - VRmin) .* rand(ps,D)) + ...
        (pos > VRmax) .* (VRmax - (VRmax - VRmin) .* rand(ps,D));

    % 引入双面镜反射边界处理
       pos = pos .* ((pos >= VRmin) & (pos <= VRmax)) + ...
        (pos < VRmin) .* (2*VRmin - pos) + ... % 反射到左侧边界
        (pos > VRmax) .* (2*VRmax - pos); % 反射到右侧边界
        
        e = feval(fhd, pos', varargin{:}); % 计算适应度
        fitcount = fitcount + ps; % 更新适应度计数
        
        % 更新个体最优
        tmp = (pbestval < e);
        temp = repmat(tmp', 1, D);
        pbest = temp .* pbest + (1 - temp) .* pos;
        pbestval = tmp .* pbestval + (1 - tmp) .* e; 
        
        % 更新全局最优
        [gbestval, tmp] = min(pbestval); 
        gbest = pbest(tmp, :);
        gbestrep = repmat(gbest, ps, 1); 
        
        % 更新收敛曲线
        convergence(i) = gbestval; 
    end
    for i = (me/2+1):me
     % 每个小组的粒子数量
         group_size = 10; % 可以根据需要调整
 
         [mean_fitness, std_fitness] = meanfs(e);
    % 更新小组最优
         [pbest_group, num_groups_actual] = update_group_best(pbest, pbestval, group_size)
    % 计算每个小组的平均适应度和标准差
          group_avg_fitness = zeros(num_groups_actual, 1);
          group_std_fitness = zeros(num_groups_actual, 1);
          for j = 1:ps/group_size
               group_indices = (j-1)*group_size+1:j*group_size;
               group_avg_fitness(j) = mean(pbestval(group_indices));
               group_std_fitness(j) = std(pbestval(group_indices));
        
          end

    % 找到符合条件的小组
    suitable_groups1 = find(group_avg_fitness > mean_fitness & group_std_fitness < std_fitness/mean_fitness);
    if isempty(suitable_groups1)
        % 对组中的每个粒子进行高斯扰动
        for j = group_indices
            % 添加高斯扰动
            perturbation_rate = 0.2; % 扰动概率
            gaussian_perturbation = rand(ps, D) < perturbation_rate;
            gaussian_sigma = 0.1; % 高斯扰动的标准差
            gaussian_vel = gaussian_sigma * randn(ps, D); % 高斯扰动
            % 根据扰动概率随机决定是否使用高斯扰动
            vel = gaussian_vel .* gaussian_perturbation + vel .* ~gaussian_perturbation;
        end
    end

    % 对符合条件的组中的最差粒子进行高斯扰动
    suitable_groups2 = find( group_std_fitness >= std_fitness  &  group_std_fitness >= std_fitness/mean_fitness);
    for g = 1:numel(suitable_groups2)
        % 找到当前组的索引范围
        group_indices = (suitable_groups2(g) - 1) * group_size + 1 : suitable_groups2(g) * group_size;
        % 找到当前组中适应度最差的粒子索引
        [~, worst_particle_idx] = max(pbestval(group_indices));
        worst_particle_idx = worst_particle_idx + (suitable_groups2(g) - 1) * group_size; % 考虑到小组的索引偏移
        % 对最差粒子进行高斯扰动
        perturbation_rate = 0.2; % 扰动概率
        gaussian_perturbation = rand(1, D) < perturbation_rate;
        gaussian_sigma = 0.1; % 高斯扰动的标准差
        gaussian_vel = gaussian_sigma * randn(1, D); % 高斯扰动
        % 根据扰动概率随机决定是否使用高斯扰动
        vel(worst_particle_idx, :) = gaussian_vel .* gaussian_perturbation + vel(worst_particle_idx, :) .* ~gaussian_perturbation;
    end
     % 对符合条件的组中的最差粒子进行高斯扰动
    suitable_groups3 = find( group_std_fitness < std_fitness  & group_std_fitness>=std_fitness/mean_fitness);
    for g = 1:numel(suitable_groups3)
        % 找到当前组的索引范围
        group_indices = (suitable_groups3(g) - 1) * group_size + 1 : suitable_groups3(g) * group_size;
        % 找到当前组中适应度最差的粒子索引
        [~, worst_particle_idx] = max(pbestval(group_indices));
        worst_particle_idx = worst_particle_idx + (suitable_groups3(g) - 1) * group_size; % 考虑到小组的索引偏移
        % 对最差粒子进行高斯扰动
        perturbation_rate = 0.2; % 扰动概率
        gaussian_perturbation = rand(1, D) < perturbation_rate;
        gaussian_sigma = 0.1; % 高斯扰动的标准差
        gaussian_vel = gaussian_sigma * randn(1, D); % 高斯扰动
        % 根据扰动概率随机决定是否使用高斯扰动
        vel(worst_particle_idx, :) = gaussian_vel .* gaussian_perturbation + vel(worst_particle_idx, :) .* ~gaussian_perturbation;
    end

        % 更新每个粒子的速度
        aa = cc1(i) .* rand(ps, D) .* (pbest - pos) + cc2(i) .* rand(ps, D) .* (gbestrep - pos)+ cc3(i) .* rand(ps, D) .* (pbest_group - pos);
        vel = iwt(i) .* vel .* gc(i) + aa; % 更新速度
        vel = (vel > Vmax) .* Vmax + (vel <= Vmax) .* vel; % 限制速度范围
        vel = (vel < Vmin) .* Vmin + (vel >= Vmin) .* vel;
      
        pos = pos + vel; % 更新位置
         % 更新粒子位置后，检查边界
        pos = pos .* ((pos >= VRmin) & (pos <= VRmax)) + (pos < VRmin) .* (VRmin + (VRmax - VRmin) .* rand(ps,D)) + ...
        (pos > VRmax) .* (VRmax - (VRmax - VRmin) .* rand(ps,D));

        % 引入双面镜反射边界处理
        pos = pos .* ((pos >= VRmin) & (pos <= VRmax)) + ...
            (pos < VRmin) .* (2*VRmin - pos) + ... % 反射到左侧边界
            (pos > VRmax) .* (2*VRmax - pos); % 反射到右侧边界
        
        e = feval(fhd, pos', varargin{:}); % 计算适应度
        fitcount = fitcount + ps; % 更新适应度计数
        
        % 更新个体最优
        tmp = (pbestval < e);
        temp = repmat(tmp', 1, D);
        pbest = temp .* pbest + (1 - temp) .* pos;
        pbestval = tmp .* pbestval + (1 - tmp) .* e; 
        
        % 更新全局最优
        [gbestval, tmp] = min(pbestval); 
        gbest = pbest(tmp, :);
        gbestrep = repmat(gbest, ps, 1); 
        
        % 更新收敛曲线
        convergence(i) = gbestval; 
    end

end

function [mean_fitness, std_fitness] = meanfs(e)
    % 计算种群适应度的平均值和标准差
    mean_fitness = mean(e);
    std_fitness = std(e);
end

function [pbest_group, num_groups_actual] = update_group_best(pbest, pbestval, group_size)
% 将粒子随机分组，并得出每个小组的最优位置
ps = size(pbest, 1); % 粒子的总数
num_groups_full = floor(ps / group_size); % 完整组数
remaining_particles = mod(ps, group_size); % 剩余粒子数
num_groups_actual = num_groups_full + (remaining_particles > 0); % 实际组数

pbest_group = zeros(ps, size(pbest, 2));

% 处理完整的小组
for g = 1:num_groups_full
    group_indices = randperm(ps, group_size);
    [~, best_idx] = min(pbestval(group_indices));
    best_particle_idx = group_indices(best_idx);
    pbest_group((g-1)*group_size+1:g*group_size, :) = repmat(pbest(best_particle_idx, :), group_size, 1);
end

% 处理剩余的粒子（如果有）
if remaining_particles > 0
    group_indices = randperm(ps, remaining_particles);
    [~, best_idx] = min(pbestval(group_indices));
    best_particle_idx = group_indices(best_idx);
    start_idx = num_groups_full * group_size + 1;
    pbest_group(start_idx:end, :) = repmat(pbest(best_particle_idx, :), remaining_particles, 1);
end
end
function sequence = golden_sequence(lower_bound, bound, n)
    phi = (1 + sqrt(5)) / 2; % 黄金分割比率
    sequence = lower_bound + bound * (1 - sin((1:n) * pi / 2 / n) * phi^(-1)); % 黄金序列生成函数
end

function [lhsPos,oblPos] = initialize_population_with_LHS_OBL(VRmin, VRmax, Particle_Number, Dimension)
    % 使用LHS初始化种群位置
    lhsDesign = lhsdesign(Particle_Number, Dimension);
    lhsPos = zeros(Particle_Number, Dimension);
    for i = 1:Dimension
        lhsPos(:, i) = VRmin(i) + (VRmax(i) - VRmin(i)) * lhsDesign(:, i);
    end
     % 应用OBL机制并计算反向位置的适应度
    [oblPos] = apply_OBL_with_perturbation(lhsPos, VRmin, VRmax);
end
function [oblPos] = apply_OBL_with_perturbation(lhsPos, VRmin, VRmax)
    % 反向学习机制
    oblPos = zeros(size(lhsPos));
    
    for i = 1:size(lhsPos, 1)
        for j = 1:size(lhsPos, 2)
            oblValue = VRmin(j) + VRmax(j) - lhsPos(i, j);
            oblPos(i, j) = oblValue; % 计算反向位置
            
        end
    end
    % 注意：这里没有直接选择较小的适应度值，因为我们在外部处理它
end

