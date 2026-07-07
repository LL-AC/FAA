
model_list=("/config/tmed/FAA_faatmed.json")
# 训练集路径
data_paths=("/TMED2/approved_users_only/DEV479/TMED2_fold0_labeledpart.csv" "/TMED2/approved_users_only/DEV479/TMED2_fold1_labeledpart.csv" "/TMED2/approved_users_only/DEV479/TMED2_fold2_labeledpart.csv")
data_folds=("0" "1" "2")
cuda=0,1
node=2
epochs=20
batch_size=64
lr=1e-3
loss=ce
mode=best_ema


# 两重循环：遍历第一个列表的每个元素，再遍历第二个列表的每个元素
for model in "${model_list[@]}"; do
    for index in "${!data_folds[@]}"; do
        fold="${data_folds[$index]}"
        echo "====================================="
        data_path=${data_paths[$index]}
        echo "开始运行：参数1=$model，参数2=${data_paths[$index]},fold=$index"

        temp=${model##*FAA_}  # 结果：random0.1.json
        # 步骤2：删除.json及之后的所有字符
        result=${temp%.json} # 结果：random0.1
        
        # 替换为你的目标脚本命令，此处以示例脚本为例
        sh test_tmed.sh $cuda $node $epochs $data_path $batch_size "largetmed" $lr $loss $mode $fold $result $model
        
        # 检查上一条命令的执行结果
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "错误：脚本运行失败！参数组合：$model 和 $data_path，退出码：$exit_code"
            # 可选：如果希望某个组合失败后停止所有运行，取消下面一行的注释
            # exit $exit_code
        fi
    done
done

echo "====================================="
echo "所有参数组合运行完毕"
