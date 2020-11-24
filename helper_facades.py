import numpy as np
import os
import glob
from PIL import Image
from scipy.misc import imresize, toimage


def preprocess_for_saving_image(im):
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    # Scale to 0-255
    #im = (((im - im.min()) * 255) / (im.max() - im.min())).astype(np.uint8)
    im = ((im + 1.0) * 127.5).astype(np.uint8)

    return im

def save_result(image_fn,
                real_image_u, g_image_u_to_v, g_image_u_to_v_to_u,
                real_image_v, g_image_v_to_u, g_image_v_to_u_to_v):
    im_0 = preprocess_for_saving_image(real_image_u)
    im_1 = preprocess_for_saving_image(g_image_u_to_v)
    im_2 = preprocess_for_saving_image(g_image_u_to_v_to_u)
    im_3 = preprocess_for_saving_image(real_image_v)
    im_4 = preprocess_for_saving_image(g_image_v_to_u)
    im_5 = preprocess_for_saving_image(g_image_v_to_u_to_v)

    concat_row_0 = np.concatenate((im_0, im_1, im_2), axis=1)
    concat_row_1 = np.concatenate((im_3, im_4, im_5), axis=1)
    concated = np.concatenate((concat_row_0, concat_row_1), axis=0)

    if concated.shape[2] == 1:
        reshaped = np.squeeze(concated, axis=2)
        toimage(reshaped, mode='L').save(image_fn)
    else:
        toimage(concated, mode='RGB').save(image_fn)

def save_result_single_row(image_fn, real_image_u, g_image_one_path, g_image_two_path):
    im_0 = preprocess_for_saving_image(real_image_u)
    im_1 = preprocess_for_saving_image(g_image_one_path)
    im_2 = preprocess_for_saving_image(g_image_two_path)
    concated = np.concatenate((im_0, im_1, im_2), axis=1)

    if concated.shape[2] == 1:
        reshaped = np.squeeze(concated, axis=2)
        toimage(reshaped, mode='L').save(image_fn)
    else:
        toimage(concated, mode='RGB').save(image_fn)



# class for loading images
class Dataset(object):
    def __init__(self, input_dir_u, input_dir_v, fn_ext, im_size, im_channel_u, im_channel_v, do_flip, do_shuffle):
        if not os.path.exists(input_dir_u) or not os.path.exists(input_dir_v):
            raise Exception('input directory does not exists!!')

        # search for images
        # self.image_files_u = glob.glob(os.path.join(input_dir_u, '*.{:s}'.format(fn_ext)))
        # self.image_files_v = glob.glob(os.path.join(input_dir_v, '*.{:s}'.format(fn_ext)))
        print("** INPUT_DIR_U**", input_dir_u )
        if 'train' in input_dir_u:
            self.image_files_u = ['A_1.jpg', 'A_10.jpg', 'A_100.jpg', 'A_101.jpg', 'A_102.jpg', 'A_103.jpg', 'A_104.jpg', 'A_105.jpg', 'A_106.jpg', 'A_107.jpg', 'A_108.jpg', 'A_109.jpg', 'A_11.jpg', 'A_110.jpg', 'A_111.jpg', 'A_112.jpg', 'A_113.jpg', 'A_114.jpg', 'A_115.jpg', 'A_116.jpg', 'A_117.jpg', 'A_118.jpg', 'A_119.jpg', 'A_12.jpg', 'A_120.jpg', 'A_121.jpg', 'A_122.jpg', 'A_123.jpg', 'A_124.jpg', 'A_125.jpg', 'A_126.jpg', 'A_127.jpg', 'A_128.jpg', 'A_129.jpg', 'A_13.jpg', 'A_130.jpg', 'A_131.jpg', 'A_132.jpg', 'A_133.jpg', 'A_134.jpg', 'A_135.jpg', 'A_136.jpg', 'A_137.jpg', 'A_138.jpg', 'A_139.jpg', 'A_14.jpg', 'A_140.jpg', 'A_141.jpg', 'A_142.jpg', 'A_143.jpg', 'A_144.jpg', 'A_145.jpg', 'A_146.jpg', 'A_147.jpg', 'A_148.jpg', 'A_149.jpg', 'A_15.jpg', 'A_150.jpg', 'A_151.jpg', 'A_152.jpg', 'A_153.jpg', 'A_154.jpg', 'A_155.jpg', 'A_156.jpg', 'A_157.jpg', 'A_158.jpg', 'A_159.jpg', 'A_16.jpg', 'A_160.jpg', 'A_161.jpg', 'A_162.jpg', 'A_163.jpg', 'A_164.jpg', 'A_165.jpg', 'A_166.jpg', 'A_167.jpg', 'A_168.jpg', 'A_169.jpg', 'A_17.jpg', 'A_170.jpg', 'A_171.jpg', 'A_172.jpg', 'A_173.jpg', 'A_174.jpg', 'A_175.jpg', 'A_176.jpg', 'A_177.jpg', 'A_178.jpg', 'A_179.jpg', 'A_18.jpg', 'A_180.jpg', 'A_181.jpg', 'A_182.jpg', 'A_183.jpg', 'A_184.jpg', 'A_185.jpg', 'A_186.jpg', 'A_187.jpg', 'A_188.jpg', 'A_189.jpg', 'A_19.jpg', 'A_190.jpg', 'A_191.jpg', 'A_192.jpg', 'A_193.jpg', 'A_194.jpg', 'A_195.jpg', 'A_196.jpg', 'A_197.jpg', 'A_198.jpg', 'A_199.jpg', 'A_2.jpg', 'A_20.jpg', 'A_200.jpg', 'A_201.jpg', 'A_202.jpg', 'A_203.jpg', 'A_204.jpg', 'A_205.jpg', 'A_206.jpg', 'A_207.jpg', 'A_208.jpg', 'A_209.jpg', 'A_21.jpg', 'A_210.jpg', 'A_211.jpg', 'A_212.jpg', 'A_213.jpg', 'A_214.jpg', 'A_215.jpg', 'A_216.jpg', 'A_217.jpg', 'A_218.jpg', 'A_219.jpg', 'A_22.jpg', 'A_220.jpg', 'A_221.jpg', 'A_222.jpg', 'A_223.jpg', 'A_224.jpg', 'A_225.jpg', 'A_226.jpg', 'A_227.jpg', 'A_228.jpg', 'A_229.jpg', 'A_23.jpg', 'A_230.jpg', 'A_231.jpg', 'A_232.jpg', 'A_233.jpg', 'A_234.jpg', 'A_235.jpg', 'A_236.jpg', 'A_237.jpg', 'A_238.jpg', 'A_239.jpg', 'A_24.jpg', 'A_240.jpg', 'A_241.jpg', 'A_242.jpg', 'A_243.jpg', 'A_244.jpg', 'A_245.jpg', 'A_246.jpg', 'A_247.jpg', 'A_248.jpg', 'A_249.jpg', 'A_25.jpg', 'A_250.jpg', 'A_251.jpg', 'A_252.jpg', 'A_253.jpg', 'A_254.jpg', 'A_255.jpg', 'A_256.jpg', 'A_257.jpg', 'A_258.jpg', 'A_259.jpg', 'A_26.jpg', 'A_260.jpg', 'A_261.jpg', 'A_262.jpg', 'A_263.jpg', 'A_264.jpg', 'A_265.jpg', 'A_266.jpg', 'A_267.jpg', 'A_268.jpg', 'A_269.jpg', 'A_27.jpg', 'A_270.jpg', 'A_271.jpg', 'A_272.jpg', 'A_273.jpg', 'A_274.jpg', 'A_275.jpg', 'A_276.jpg', 'A_277.jpg', 'A_278.jpg', 'A_279.jpg', 'A_28.jpg', 'A_280.jpg', 'A_281.jpg', 'A_282.jpg', 'A_283.jpg', 'A_284.jpg', 'A_285.jpg', 'A_286.jpg', 'A_287.jpg', 'A_288.jpg', 'A_289.jpg', 'A_29.jpg', 'A_290.jpg', 'A_291.jpg', 'A_292.jpg', 'A_293.jpg', 'A_294.jpg', 'A_295.jpg', 'A_296.jpg', 'A_297.jpg', 'A_298.jpg', 'A_299.jpg', 'A_3.jpg', 'A_30.jpg', 'A_300.jpg', 'A_301.jpg', 'A_302.jpg', 'A_303.jpg', 'A_304.jpg', 'A_305.jpg', 'A_306.jpg', 'A_307.jpg', 'A_308.jpg', 'A_309.jpg', 'A_31.jpg', 'A_310.jpg', 'A_311.jpg', 'A_312.jpg', 'A_313.jpg', 'A_314.jpg', 'A_315.jpg', 'A_316.jpg', 'A_317.jpg', 'A_318.jpg', 'A_319.jpg', 'A_32.jpg', 'A_320.jpg', 'A_321.jpg', 'A_322.jpg', 'A_323.jpg', 'A_324.jpg', 'A_325.jpg', 'A_326.jpg', 'A_327.jpg', 'A_328.jpg', 'A_329.jpg', 'A_33.jpg', 'A_330.jpg', 'A_331.jpg', 'A_332.jpg', 'A_333.jpg', 'A_334.jpg', 'A_335.jpg', 'A_336.jpg', 'A_337.jpg', 'A_338.jpg', 'A_339.jpg', 'A_34.jpg', 'A_340.jpg', 'A_341.jpg', 'A_342.jpg', 'A_343.jpg', 'A_344.jpg', 'A_345.jpg', 'A_346.jpg', 'A_347.jpg', 'A_348.jpg', 'A_349.jpg', 'A_35.jpg', 'A_350.jpg', 'A_351.jpg', 'A_352.jpg', 'A_353.jpg', 'A_354.jpg', 'A_355.jpg', 'A_356.jpg', 'A_357.jpg', 'A_358.jpg', 'A_359.jpg', 'A_36.jpg', 'A_360.jpg', 'A_361.jpg', 'A_362.jpg', 'A_363.jpg', 'A_364.jpg', 'A_365.jpg', 'A_366.jpg', 'A_367.jpg', 'A_368.jpg', 'A_369.jpg', 'A_37.jpg', 'A_370.jpg', 'A_371.jpg', 'A_372.jpg', 'A_373.jpg', 'A_374.jpg', 'A_375.jpg', 'A_376.jpg', 'A_377.jpg', 'A_378.jpg', 'A_379.jpg', 'A_38.jpg', 'A_380.jpg', 'A_381.jpg', 'A_382.jpg', 'A_383.jpg', 'A_384.jpg', 'A_385.jpg', 'A_386.jpg', 'A_387.jpg', 'A_388.jpg', 'A_389.jpg', 'A_39.jpg', 'A_390.jpg', 'A_391.jpg', 'A_392.jpg', 'A_393.jpg', 'A_394.jpg', 'A_395.jpg', 'A_396.jpg', 'A_397.jpg', 'A_398.jpg', 'A_399.jpg', 'A_4.jpg', 
'A_40.jpg', 'A_400.jpg', 'A_41.jpg', 'A_42.jpg', 'A_43.jpg', 'A_44.jpg', 'A_45.jpg', 'A_46.jpg', 'A_47.jpg', 'A_48.jpg', 'A_49.jpg', 'A_5.jpg', 'A_50.jpg', 'A_51.jpg', 'A_52.jpg', 'A_53.jpg', 'A_54.jpg', 'A_55.jpg', 'A_56.jpg', 'A_57.jpg', 'A_58.jpg', 'A_59.jpg', 'A_6.jpg', 'A_60.jpg', 'A_61.jpg', 'A_62.jpg', 'A_63.jpg', 'A_64.jpg', 'A_65.jpg', 'A_66.jpg', 'A_67.jpg', 'A_68.jpg', 'A_69.jpg', 'A_7.jpg', 'A_70.jpg', 'A_71.jpg', 'A_72.jpg', 'A_73.jpg', 'A_74.jpg', 'A_75.jpg', 'A_76.jpg', 'A_77.jpg', 'A_78.jpg', 'A_79.jpg', 'A_8.jpg', 'A_80.jpg', 'A_81.jpg', 'A_82.jpg', 'A_83.jpg', 'A_84.jpg', 'A_85.jpg', 'A_86.jpg', 'A_87.jpg', 'A_88.jpg', 'A_89.jpg', 'A_9.jpg', 'A_90.jpg', 'A_91.jpg', 'A_92.jpg', 'A_93.jpg', 'A_94.jpg', 'A_95.jpg', 'A_96.jpg', 'A_97.jpg', 'A_98.jpg', 'A_99.jpg']
            self.image_files_v = ['B_1.jpg', 'B_10.jpg', 'B_100.jpg', 'B_101.jpg', 'B_102.jpg', 'B_103.jpg', 'B_104.jpg', 'B_105.jpg', 'B_106.jpg', 'B_107.jpg', 'B_108.jpg', 'B_109.jpg', 'B_11.jpg', 'B_110.jpg', 'B_111.jpg', 'B_112.jpg', 'B_113.jpg', 'B_114.jpg', 'B_115.jpg', 'B_116.jpg', 'B_117.jpg', 'B_118.jpg', 'B_119.jpg', 'B_12.jpg', 'B_120.jpg', 'B_121.jpg', 'B_122.jpg', 'B_123.jpg', 'B_124.jpg', 'B_125.jpg', 'B_126.jpg', 'B_127.jpg', 'B_128.jpg', 'B_129.jpg', 'B_13.jpg', 'B_130.jpg', 'B_131.jpg', 'B_132.jpg', 'B_133.jpg', 'B_134.jpg', 'B_135.jpg', 'B_136.jpg', 'B_137.jpg', 'B_138.jpg', 'B_139.jpg', 'B_14.jpg', 'B_140.jpg', 'B_141.jpg', 'B_142.jpg', 'B_143.jpg', 'B_144.jpg', 'B_145.jpg', 'B_146.jpg', 'B_147.jpg', 'B_148.jpg', 'B_149.jpg', 'B_15.jpg', 'B_150.jpg', 'B_151.jpg', 'B_152.jpg', 'B_153.jpg', 'B_154.jpg', 'B_155.jpg', 'B_156.jpg', 'B_157.jpg', 'B_158.jpg', 'B_159.jpg', 'B_16.jpg', 'B_160.jpg', 'B_161.jpg', 'B_162.jpg', 'B_163.jpg', 'B_164.jpg', 'B_165.jpg', 'B_166.jpg', 'B_167.jpg', 'B_168.jpg', 'B_169.jpg', 'B_17.jpg', 'B_170.jpg', 'B_171.jpg', 'B_172.jpg', 'B_173.jpg', 'B_174.jpg', 'B_175.jpg', 'B_176.jpg', 'B_177.jpg', 'B_178.jpg', 'B_179.jpg', 'B_18.jpg', 'B_180.jpg', 'B_181.jpg', 'B_182.jpg', 'B_183.jpg', 'B_184.jpg', 'B_185.jpg', 'B_186.jpg', 'B_187.jpg', 'B_188.jpg', 'B_189.jpg', 'B_19.jpg', 'B_190.jpg', 'B_191.jpg', 'B_192.jpg', 'B_193.jpg', 'B_194.jpg', 'B_195.jpg', 'B_196.jpg', 'B_197.jpg', 'B_198.jpg', 'B_199.jpg', 'B_2.jpg', 'B_20.jpg', 'B_200.jpg', 'B_201.jpg', 'B_202.jpg', 'B_203.jpg', 'B_204.jpg', 'B_205.jpg', 'B_206.jpg', 'B_207.jpg', 'B_208.jpg', 'B_209.jpg', 'B_21.jpg', 'B_210.jpg', 'B_211.jpg', 'B_212.jpg', 'B_213.jpg', 'B_214.jpg', 'B_215.jpg', 'B_216.jpg', 'B_217.jpg', 'B_218.jpg', 'B_219.jpg', 'B_22.jpg', 'B_220.jpg', 'B_221.jpg', 'B_222.jpg', 'B_223.jpg', 'B_224.jpg', 'B_225.jpg', 'B_226.jpg', 'B_227.jpg', 'B_228.jpg', 'B_229.jpg', 'B_23.jpg', 'B_230.jpg', 'B_231.jpg', 'B_232.jpg', 'B_233.jpg', 'B_234.jpg', 'B_235.jpg', 'B_236.jpg', 'B_237.jpg', 'B_238.jpg', 'B_239.jpg', 'B_24.jpg', 'B_240.jpg', 'B_241.jpg', 'B_242.jpg', 'B_243.jpg', 'B_244.jpg', 'B_245.jpg', 'B_246.jpg', 'B_247.jpg', 'B_248.jpg', 'B_249.jpg', 'B_25.jpg', 'B_250.jpg', 'B_251.jpg', 'B_252.jpg', 'B_253.jpg', 'B_254.jpg', 'B_255.jpg', 'B_256.jpg', 'B_257.jpg', 'B_258.jpg', 'B_259.jpg', 'B_26.jpg', 'B_260.jpg', 'B_261.jpg', 'B_262.jpg', 'B_263.jpg', 'B_264.jpg', 'B_265.jpg', 'B_266.jpg', 'B_267.jpg', 'B_268.jpg', 'B_269.jpg', 'B_27.jpg', 'B_270.jpg', 'B_271.jpg', 'B_272.jpg', 'B_273.jpg', 'B_274.jpg', 'B_275.jpg', 'B_276.jpg', 'B_277.jpg', 'B_278.jpg', 'B_279.jpg', 'B_28.jpg', 'B_280.jpg', 'B_281.jpg', 'B_282.jpg', 'B_283.jpg', 'B_284.jpg', 'B_285.jpg', 'B_286.jpg', 'B_287.jpg', 'B_288.jpg', 'B_289.jpg', 'B_29.jpg', 'B_290.jpg', 'B_291.jpg', 'B_292.jpg', 'B_293.jpg', 'B_294.jpg', 'B_295.jpg', 'B_296.jpg', 'B_297.jpg', 'B_298.jpg', 'B_299.jpg', 'B_3.jpg', 'B_30.jpg', 'B_300.jpg', 'B_301.jpg', 'B_302.jpg', 'B_303.jpg', 'B_304.jpg', 'B_305.jpg', 'B_306.jpg', 'B_307.jpg', 'B_308.jpg', 'B_309.jpg', 'B_31.jpg', 'B_310.jpg', 'B_311.jpg', 'B_312.jpg', 'B_313.jpg', 'B_314.jpg', 'B_315.jpg', 'B_316.jpg', 'B_317.jpg', 'B_318.jpg', 'B_319.jpg', 'B_32.jpg', 'B_320.jpg', 'B_321.jpg', 'B_322.jpg', 'B_323.jpg', 'B_324.jpg', 'B_325.jpg', 'B_326.jpg', 'B_327.jpg', 'B_328.jpg', 'B_329.jpg', 'B_33.jpg', 'B_330.jpg', 'B_331.jpg', 'B_332.jpg', 'B_333.jpg', 'B_334.jpg', 'B_335.jpg', 'B_336.jpg', 'B_337.jpg', 'B_338.jpg', 'B_339.jpg', 'B_34.jpg', 'B_340.jpg', 'B_341.jpg', 'B_342.jpg', 'B_343.jpg', 'B_344.jpg', 'B_345.jpg', 'B_346.jpg', 'B_347.jpg', 'B_348.jpg', 'B_349.jpg', 'B_35.jpg', 'B_350.jpg', 'B_351.jpg', 'B_352.jpg', 'B_353.jpg', 'B_354.jpg', 'B_355.jpg', 'B_356.jpg', 'B_357.jpg', 'B_358.jpg', 'B_359.jpg', 'B_36.jpg', 'B_360.jpg', 'B_361.jpg', 'B_362.jpg', 'B_363.jpg', 'B_364.jpg', 'B_365.jpg', 'B_366.jpg', 'B_367.jpg', 'B_368.jpg', 'B_369.jpg', 'B_37.jpg', 'B_370.jpg', 'B_371.jpg', 'B_372.jpg', 'B_373.jpg', 'B_374.jpg', 'B_375.jpg', 'B_376.jpg', 'B_377.jpg', 'B_378.jpg', 'B_379.jpg', 'B_38.jpg', 'B_380.jpg', 'B_381.jpg', 'B_382.jpg', 'B_383.jpg', 'B_384.jpg', 'B_385.jpg', 'B_386.jpg', 'B_387.jpg', 'B_388.jpg', 'B_389.jpg', 'B_39.jpg', 'B_390.jpg', 'B_391.jpg', 'B_392.jpg', 'B_393.jpg', 'B_394.jpg', 'B_395.jpg', 'B_396.jpg', 'B_397.jpg', 'B_398.jpg', 'B_399.jpg', 'B_4.jpg', 
'B_40.jpg', 'B_400.jpg', 'B_41.jpg', 'B_42.jpg', 'B_43.jpg', 'B_44.jpg', 'B_45.jpg', 'B_46.jpg', 'B_47.jpg', 'B_48.jpg', 'B_49.jpg', 'B_5.jpg', 'B_50.jpg', 'B_51.jpg', 'B_52.jpg', 'B_53.jpg', 'B_54.jpg', 'B_55.jpg', 'B_56.jpg', 'B_57.jpg', 'B_58.jpg', 'B_59.jpg', 'B_6.jpg', 'B_60.jpg', 'B_61.jpg', 'B_62.jpg', 'B_63.jpg', 'B_64.jpg', 'B_65.jpg', 'B_66.jpg', 'B_67.jpg', 'B_68.jpg', 'B_69.jpg', 'B_7.jpg', 'B_70.jpg', 'B_71.jpg', 'B_72.jpg', 'B_73.jpg', 'B_74.jpg', 'B_75.jpg', 'B_76.jpg', 'B_77.jpg', 'B_78.jpg', 'B_79.jpg', 'B_8.jpg', 'B_80.jpg', 'B_81.jpg', 'B_82.jpg', 'B_83.jpg', 'B_84.jpg', 'B_85.jpg', 'B_86.jpg', 'B_87.jpg', 'B_88.jpg', 'B_89.jpg', 'B_9.jpg', 'B_90.jpg', 'B_91.jpg', 'B_92.jpg', 'B_93.jpg', 'B_94.jpg', 'B_95.jpg', 'B_96.jpg', 'B_97.jpg', 'B_98.jpg', 'B_99.jpg']
        else:
            self.image_files_u = ['A_1.jpg', 'A_10.jpg', 'A_100.jpg', 'A_11.jpg', 'A_12.jpg', 'A_13.jpg', 'A_14.jpg', 'A_15.jpg', 'A_16.jpg', 'A_17.jpg', 'A_18.jpg', 'A_19.jpg', 'A_2.jpg', 'A_20.jpg', 'A_21.jpg', 'A_22.jpg', 'A_23.jpg', 'A_24.jpg', 'A_25.jpg', 'A_26.jpg', 'A_27.jpg', 'A_28.jpg', 'A_29.jpg', 'A_3.jpg', 'A_30.jpg', 'A_31.jpg', 'A_32.jpg', 'A_33.jpg', 'A_34.jpg', 'A_35.jpg', 'A_36.jpg', 'A_37.jpg', 'A_38.jpg', 'A_39.jpg', 'A_4.jpg', 'A_40.jpg', 'A_41.jpg', 'A_42.jpg', 'A_43.jpg', 'A_44.jpg', 'A_45.jpg', 'A_46.jpg', 'A_47.jpg', 'A_48.jpg', 'A_49.jpg', 'A_5.jpg', 'A_50.jpg', 'A_51.jpg', 'A_52.jpg', 'A_53.jpg', 'A_54.jpg', 'A_55.jpg', 'A_56.jpg', 'A_57.jpg', 'A_58.jpg', 'A_59.jpg', 'A_6.jpg', 'A_60.jpg', 'A_61.jpg', 'A_62.jpg', 'A_63.jpg', 'A_64.jpg', 'A_65.jpg', 'A_66.jpg', 'A_67.jpg', 'A_68.jpg', 'A_69.jpg', 'A_7.jpg', 'A_70.jpg', 'A_71.jpg', 'A_72.jpg', 'A_73.jpg', 'A_74.jpg', 'A_75.jpg', 'A_76.jpg', 'A_77.jpg', 'A_78.jpg', 'A_79.jpg', 'A_8.jpg', 'A_80.jpg', 'A_81.jpg', 'A_82.jpg', 'A_83.jpg', 'A_84.jpg', 'A_85.jpg', 'A_86.jpg', 'A_87.jpg', 'A_88.jpg', 'A_89.jpg', 'A_9.jpg', 'A_90.jpg', 'A_91.jpg', 'A_92.jpg', 'A_93.jpg', 'A_94.jpg', 'A_95.jpg', 'A_96.jpg', 'A_97.jpg', 'A_98.jpg', 'A_99.jpg']
            self.image_files_v = ['B_1.jpg', 'B_10.jpg', 'B_100.jpg', 'B_11.jpg', 'B_12.jpg', 'B_13.jpg', 'B_14.jpg', 'B_15.jpg', 'B_16.jpg', 'B_17.jpg', 'B_18.jpg', 'B_19.jpg', 'B_2.jpg', 'B_20.jpg', 'B_21.jpg', 'B_22.jpg', 'B_23.jpg', 'B_24.jpg', 'B_25.jpg', 'B_26.jpg', 'B_27.jpg', 'B_28.jpg', 'B_29.jpg', 'B_3.jpg', 'B_30.jpg', 'B_31.jpg', 'B_32.jpg', 'B_33.jpg', 'B_34.jpg', 'B_35.jpg', 'B_36.jpg', 'B_37.jpg', 'B_38.jpg', 'B_39.jpg', 'B_4.jpg', 'B_40.jpg', 'B_41.jpg', 'B_42.jpg', 'B_43.jpg', 'B_44.jpg', 'B_45.jpg', 'B_46.jpg', 'B_47.jpg', 'B_48.jpg', 'B_49.jpg', 'B_5.jpg', 'B_50.jpg', 'B_51.jpg', 'B_52.jpg', 'B_53.jpg', 'B_54.jpg', 'B_55.jpg', 'B_56.jpg', 'B_57.jpg', 'B_58.jpg', 'B_59.jpg', 'B_6.jpg', 'B_60.jpg', 'B_61.jpg', 'B_62.jpg', 'B_63.jpg', 'B_64.jpg', 'B_65.jpg', 'B_66.jpg', 'B_67.jpg', 'B_68.jpg', 'B_69.jpg', 'B_7.jpg', 'B_70.jpg', 'B_71.jpg', 'B_72.jpg', 'B_73.jpg', 'B_74.jpg', 'B_75.jpg', 'B_76.jpg', 'B_77.jpg', 'B_78.jpg', 'B_79.jpg', 'B_8.jpg', 'B_80.jpg', 'B_81.jpg', 'B_82.jpg', 'B_83.jpg', 'B_84.jpg', 'B_85.jpg', 'B_86.jpg', 'B_87.jpg', 'B_88.jpg', 'B_89.jpg', 'B_9.jpg', 'B_90.jpg', 'B_91.jpg', 'B_92.jpg', 'B_93.jpg', 'B_94.jpg', 'B_95.jpg', 'B_96.jpg', 'B_97.jpg', 'B_98.jpg', 'B_99.jpg']
        for i, path in enumerate(self.image_files_u):
            self.image_files_u[i] = input_dir_u + self.image_files_u[i]

        for i, path in enumerate(self.image_files_v):
            self.image_files_v[i] = input_dir_v + self.image_files_v[i]


        # for entry in u_files:
        #     fullPath = os.path.join(input_dir_u, entry)
        #     self.image_files_u.append(fullPath)
        # for entry in v_files:
        #     fullPath = os.path.join(input_dir_v, entry)
        #     self.image_files_v.append(fullPath)

        # self.image_files_u = ['/content/datasets/day-night/train/A/A_6819.jpg', '/content/datasets/day-night/train/A/A_6728.jpg', '/content/datasets/day-night/train/A/A_6961.jpg', '/content/datasets/day-night/train/A/A_7627.jpg']
        # self.image_files_v = ['/content/datasets/day-night/train/B/B_6819.jpg', '/content/datasets/day-night/train/B/B_6728.jpg', '/content/datasets/day-night/train/B/B_6961.jpg', '/content/datasets/day-night/train/B/B_7627.jpg']

        print("** INSIDE INIT **")
        print(self.image_files_u)
        print(self.image_files_v)

        if len(self.image_files_u) == 0 or len(self.image_files_v) == 0:
            raise Exception('input directory does not contain any images!!')

        # shuffle image files
        self.do_shuffle = do_shuffle
        if self.do_shuffle:
            # print(glob.glob(os.path.join('./day-night/train/A', '*.{:s}'.format('jpg'))))
            pass
            np.random.seed(0)
            np.random.shuffle(self.image_files_u)
            np.random.seed(0)
            np.random.shuffle(self.image_files_v)

        self.n_images = len(self.image_files_u) if len(self.image_files_u) <= len(self.image_files_v) else len(self.image_files_v)
        self.batch_index = 0
        self.resize_to = im_size
        self.color_mode_u = 'L' if im_channel_u == 1 else 'RGB'
        self.color_mode_v = 'L' if im_channel_v == 1 else 'RGB'
        self.do_flip = do_flip
        self.image_max_value = 255
        self.image_max_value_half = 127.5
        # self.prng = np.random.RandomState(777)

    def reset(self):
        self.batch_index = 0
        # shuffle image files
        if self.do_shuffle:
            np.random.shuffle(self.image_files_u)
            np.random.shuffle(self.image_files_v)

    def get_image_by_index(self, index):
        if index >= self.n_images:
            index = 0

        print(self.image_files_u)
        print(self.image_files_v)

        fn_u = [self.image_files_u[index]]
        fn_v = [self.image_files_v[index]]
        print("** INSIDE HELPER**")
        print("*** FN u ***", fn_u)
        print("*** FN v ***", fn_v)

        image_u = self.load_image(fn_u, self.color_mode_u)
        image_v = self.load_image(fn_v, self.color_mode_v)
        return image_u, image_v

    def get_image_by_index_u(self, index):
        if index >= self.n_images:
            index = 0

        fn_u = [self.image_files_u[index]]
        image_u = self.load_image(fn_u, self.color_mode_u)
        return image_u

    def get_image_by_index_v(self, index):
        if index >= self.n_images:
            index = 0

        fn_v = [self.image_files_v[index]]
        image_v = self.load_image(fn_v, self.color_mode_v)
        return image_v

    def get_next_batch(self, batch_size):
        if (self.batch_index + batch_size) > self.n_images:
            self.batch_index = 0

        batch_files_u = self.image_files_u[self.batch_index:self.batch_index + batch_size]
        batch_files_v = self.image_files_v[self.batch_index:self.batch_index + batch_size]

        images_u = self.load_image(batch_files_u, self.color_mode_u)
        images_v = self.load_image(batch_files_v, self.color_mode_v)

        self.batch_index += batch_size

        return images_u, images_v

    def load_image(self, fn_list, color_mode):
        images = []
        for fn in fn_list:
            # open images with PIL
            im = Image.open(fn)
            im = np.array(im.convert(color_mode))

            # resize
            im = imresize(im, [self.resize_to, self.resize_to])

            # perform flip if needed
            # random_val = self.prng.uniform(0, 1)
            random_val = np.random.random()
            if self.do_flip and random_val > 0.5:
                #im = np.flip(im, axis=1)
                im = np.fliplr(im)

            # normalize input [0 ~ 255] ==> [-1 ~ 1]
            #im = (im / self.image_max_value - 0.5) * 2
            im = im / self.image_max_value_half - 1.0

            # make 3 dimensional for single channel image
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            images.append(im)
        images = np.array(images)

        return images