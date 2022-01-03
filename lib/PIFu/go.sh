

python -m apps.prt_util \
-i ./data/rp_dennis_posed_004_OBJ/

python -m apps.render_data \
-i ./data/rp_dennis_posed_004_OBJ/ \
-o ./data/training/ \
-e


python -m apps.train_shape --dataroot /home/hj/__workspace__/PIFu/data/training/ --random_flip --random_scale --random_trans