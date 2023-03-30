import os
import shutil

results_dir = './data_split/test'
target_dir = 'E:/VS_Projects/Explorer/data'

def collect_imgs():
    os.makedirs(target_dir, exist_ok=True)
    t=0
    while True:
        scene_name = f'{results_dir}/{t}_scene.png'
        if not os.path.exists(scene_name):
            break
        mask_name = f'{results_dir}/{t}_mask.png'
        normal_name = f'{results_dir}/results/{t}_scene_pred_norm.png'
        uncertainty_name = f'{results_dir}/results/{t}_scene_pred_alpha.png'

        shutil.copy2(scene_name, target_dir)
        shutil.copy2(mask_name, target_dir)
        shutil.copy2(normal_name, target_dir)
        shutil.copy2(uncertainty_name, target_dir)

        t += 1
    print(f'Handled {t} pictures.')
    pass

if __name__ == '__main__':
    collect_imgs()
    pass