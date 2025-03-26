1.启动模型：
点击starthyj.bat即可，启动模型和音色配置会自动在后台运行。当前使用的音色模型可以在gpt终端中看到。as：
t2s_weights_path    : C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\meidusha-e10.ckpt
vits_weights_path   : C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\meidusha_e10_s210.pth


2.如需更换音色：
2.1在浏览器粘贴以下要选择的音色（直接粘贴网址），网页显示success就是成功使用。
2.2在conf.yaml中更换相关音色配置文件wav及提示词。(这里不给会报错，但是理论上可以不给，代码方面可以寻找修改，这样更改音色更为简单，但是不知道效果是否下降)

音色1：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\ganyu-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\ganyu_e10_s170.pth

音色2：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\hutao-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\hutao_e10_s210.pth

音色3：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\linhua-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\linhua_e10_s820.pth

音色4：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\meidusha-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\meidusha_e10_s210.pth

音色5：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\diaona-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\diaona_e10_s210.pth

音色6：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\babala-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\babala_e10_s220.pth

音色7：有问题
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\fufu-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\fufu_e10_s290.pth

音色8：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\naxida-e10.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\naxida_e10_s400.pth

音色9：
http://127.0.0.1:9880/set_gpt_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\GPT_weights_v2\000-e30.ckpt
http://127.0.0.1:9880/set_sovits_weights?weights_path=C:\Users\20222\yj-VTuber\GPT-SoVITS-v2\SoVITS_weights_v2\000_e24_s1104.pth