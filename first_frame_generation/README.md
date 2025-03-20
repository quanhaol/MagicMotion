# First Frame Generation

We provide scripts here that allow users to use FLUX to generate or edit an image to serve as the input image for MagicMotion.
Please refer to `edit_image_flux.py`  and `t2i_flux.py` for more details.

```bash
# For text-to-image generation
python first_frame_generation/t2i_flux.py --prompt "A tiger sitting, head facing camera." --save_path "assets/images/condition/tiger.png"
# For Image Editing
python first_frame_generation/edit_image_flux.py --prompt "A royal camel walking inside a palace." --ori_image_path "assets/images/ori_image/camel.jpg" --save_path "assets/images/condition/camel_royal.png"
```
