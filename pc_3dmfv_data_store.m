function [pcds] = pc_3dmfv_data_store(path, n_points, GMM, normalize, flatten, is_training, augmentations)
% pc_3dmfv_data_store reutrns a point cloud data store for labled point 返回标记点云的点云数据库
% clouds. It requires the directory names to be the labels.值得注意的是目录名必须是标签
%INPUT: path - string containing the path to directory which contains the labled subdirectories of point
%clouds
%OUTPUT: pcds - an image datastore object of 3dmfv representaiton 一个3dmfv表示的图像数据存储对象

pcds = imageDatastore(path,...
    'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

    function pc_3dmfv = pc_reader(filename)
        points = table2array(readtable(filename));
        points = points(randperm(n_points),:);
        points = Shrink2UnitSphere(gpuArray(points)); %将点放缩到单位球当中去
        % Maybe add augmentaitons in trainingset
        if is_training
            points = augment_data(gpuArray(points), augmentations); %加入噪声
        end
        pc_3dmfv = compute_3dmfv(gpuArray(points), gpuArray(GMM.w), gpuArray(GMM.mu), gpuArray(GMM.sigma), gpuArray(normalize), gpuArray(flatten));
    end
end


function [newPoints]=Shrink2UnitSphere(Points)
%Shrink2UnitSphere shrinks the given data x,y,z Points to fit insed a unit sphere
%and returns the new dataset
%INPUT : Points nx3 输入点的三个坐标，返回新的点的坐标
% move model to center of gepmetry 
xyzmean=mean(Points, 1);
newPoints(:, 1)=Points(:, 1)-xyzmean(1);
newPoints(:, 2)=Points(:, 2)-xyzmean(2);
newPoints(:, 3)=Points(:, 3)-xyzmean(3);
%Shring the model
dist= sqrt(sum(newPoints.^2,2));
maxdist=max(dist);
newPoints=newPoints/(maxdist);
end

function augmented_points = augment_data(points, augmentations)
% Insert data augmentations 新加入一些点作为干扰，用于后续的实验
augmented_points = points;

if augmentations(1)
    augmented_points = rotate_point_cloud(points); %围绕y轴插入随机旋转
end
if augmentations(2)
    augmented_points = scale_point_cloud(augmented_points, [0.66, 1.5]); %随机各向异性尺度
end
if augmentations(3) 
    augmented_points = translate_point_cloud(points, 0.2); %在给定的范围内随机平移
end
if augmentations(4) 
    augmented_points = jitter_point_cloud(augmented_points, 0.01, 0.05); %加入高斯噪声
end
if augmentations(5) 
    augmented_points = insert_outliers_to_point_cloud(points, 0.05); %插入异常噪声
end
end

function scaled_points = scale_point_cloud(points, s_range)
% random anisotropic scale of point clouds within a given range 给定范围内点云的随机各向异性尺度
smin = s_range(1);
smax = s_range(2);
s = (smax - smin).*rand(3, 1,'gpuArray') + smin;
scale_matrix = gpuArray([s(1), 0, 0; 0, s(2), 0; 0, 0, s(3)]);
scaled_points = scale_matrix(1:3, 1:3) * points';
scaled_points = scaled_points';
end

function translated_points = translate_point_cloud(points, t_val)
% random translation of point clouds within a given range 点云在给定范围内的随机平移

translation = 2*t_val.*rand(1, 3,'gpuArray') - t_val;
translation = repmat(translation,size(points,1), 1);
translated_points = points + translation;
end

function jittered_points = jitter_point_cloud(points, sigma, clip)
% Insert Gaussian noise 加入高斯噪声

noise = sigma * randn(size(points),'gpuArray');
noise(noise > clip) = clip;
noise(noise < -clip) = -clip; 
jittered_points = points + noise;
end

function outlier_data = insert_outliers_to_point_cloud(points, outlier_ratio)
% Insert outlier noise 插入异常噪声
n_points = size(points, 1);
n_outliers =uint64(outlier_ratio * n_points);

outlier_data = points;
idx = randperm(n_points, n_outliers);
outlier_data(idx,:) = 2 * randn([n_outliers, 3],'gpuArray') - 1;
end

function rotated_data = rotate_point_cloud(points)
% Insert random rotations around y axis 围绕y轴插入随机旋转
angle = 2*pi*randn(1);
cos_a = cos(angle);
sin_a = sin(angle); 
Ry = [cos_a, 0, sin_a; 0, 1, 0; -sin_a, 0, cos_a];
rotated_data = Ry * points';
rotated_data = rotated_data';
end
