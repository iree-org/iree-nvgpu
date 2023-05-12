// RUN: iree-compile %s --iree-plugin=openxla_nvgpu --iree-input-type=mhlo     \
// RUN:     --iree-hal-target-backends=cuda                                    \
// RUN: | iree-run-module --module=- --device=cuda --function=predict          \
// RUN:     --input=1x224x224x3xf32                                            \
// RUN: FileCheck %s


// CHECK: EXEC @predict
// CHECK: 1x1000xf32
// CHECK: [0.001 0.001 0.001 0.001 0.001

util.global private @"__iree_flow___sm_node188__m.layer-2.kernel" {noinline} = dense<1.000000e+00> : tensor<7x7x3x64xf32>
util.global private @"__iree_flow___sm_node189__m.layer-2.bias" {noinline} = dense<5.000000e-01> : tensor<64xf32>
util.global private @"__iree_flow___sm_node195__m.layer-3.gamma" {noinline} = dense<0.333333343> : tensor<64xf32>
util.global private @"__iree_flow___sm_node196__m.layer-3.beta" {noinline} = dense<2.500000e-01> : tensor<64xf32>
util.global private @"__iree_flow___sm_node197__m.layer-3.moving_mean" {noinline} = dense<2.000000e-01> : tensor<64xf32>
util.global private @"__iree_flow___sm_node198__m.layer-3.moving_variance" {noinline} = dense<0.166666672> : tensor<64xf32>
util.global private @"__iree_flow___sm_node215__m.layer-7.kernel" {noinline} = dense<0.142857149> : tensor<1x1x64x64xf32>
util.global private @"__iree_flow___sm_node216__m.layer-7.bias" {noinline} = dense<1.250000e-01> : tensor<64xf32>
util.global private @"__iree_flow___sm_node222__m.layer-8.gamma" {noinline} = dense<0.111111112> : tensor<64xf32>
util.global private @"__iree_flow___sm_node223__m.layer-8.beta" {noinline} = dense<1.000000e-01> : tensor<64xf32>
util.global private @"__iree_flow___sm_node224__m.layer-8.moving_mean" {noinline} = dense<0.0909090936> : tensor<64xf32>
util.global private @"__iree_flow___sm_node225__m.layer-8.moving_variance" {noinline} = dense<0.0833333358> : tensor<64xf32>
util.global private @"__iree_flow___sm_node234__m.layer-10.kernel" {noinline} = dense<0.0769230798> : tensor<3x3x64x64xf32>
util.global private @"__iree_flow___sm_node235__m.layer-10.bias" {noinline} = dense<0.0714285746> : tensor<64xf32>
util.global private @"__iree_flow___sm_node241__m.layer-11.gamma" {noinline} = dense<0.0666666701> : tensor<64xf32>
util.global private @"__iree_flow___sm_node242__m.layer-11.beta" {noinline} = dense<6.250000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node243__m.layer-11.moving_mean" {noinline} = dense<0.0588235296> : tensor<64xf32>
util.global private @"__iree_flow___sm_node244__m.layer-11.moving_variance" {noinline} = dense<0.055555556> : tensor<64xf32>
util.global private @"__iree_flow___sm_node253__m.layer-13.kernel" {noinline} = dense<0.0526315793> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node254__m.layer-13.bias" {noinline} = dense<5.000000e-02> : tensor<256xf32>
util.global private @"__iree_flow___sm_node259__m.layer-14.kernel" {noinline} = dense<0.0476190485> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node260__m.layer-14.bias" {noinline} = dense<0.0454545468> : tensor<256xf32>
util.global private @"__iree_flow___sm_node266__m.layer-15.gamma" {noinline} = dense<0.0434782617> : tensor<256xf32>
util.global private @"__iree_flow___sm_node267__m.layer-15.beta" {noinline} = dense<0.0416666679> : tensor<256xf32>
util.global private @"__iree_flow___sm_node268__m.layer-15.moving_mean" {noinline} = dense<4.000000e-02> : tensor<256xf32>
util.global private @"__iree_flow___sm_node269__m.layer-15.moving_variance" {noinline} = dense<0.0384615399> : tensor<256xf32>
util.global private @"__iree_flow___sm_node275__m.layer-16.gamma" {noinline} = dense<0.0370370373> : tensor<256xf32>
util.global private @"__iree_flow___sm_node276__m.layer-16.beta" {noinline} = dense<0.0357142873> : tensor<256xf32>
util.global private @"__iree_flow___sm_node277__m.layer-16.moving_mean" {noinline} = dense<0.0344827585> : tensor<256xf32>
util.global private @"__iree_flow___sm_node278__m.layer-16.moving_variance" {noinline} = dense<0.0333333351> : tensor<256xf32>
util.global private @"__iree_flow___sm_node291__m.layer-19.kernel" {noinline} = dense<0.0322580636> : tensor<1x1x256x64xf32>
util.global private @"__iree_flow___sm_node292__m.layer-19.bias" {noinline} = dense<3.125000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node298__m.layer-20.gamma" {noinline} = dense<0.0303030312> : tensor<64xf32>
util.global private @"__iree_flow___sm_node299__m.layer-20.beta" {noinline} = dense<0.0294117648> : tensor<64xf32>
util.global private @"__iree_flow___sm_node300__m.layer-20.moving_mean" {noinline} = dense<0.0285714287> : tensor<64xf32>
util.global private @"__iree_flow___sm_node301__m.layer-20.moving_variance" {noinline} = dense<0.027777778> : tensor<64xf32>
util.global private @"__iree_flow___sm_node310__m.layer-22.kernel" {noinline} = dense<0.0270270277> : tensor<3x3x64x64xf32>
util.global private @"__iree_flow___sm_node311__m.layer-22.bias" {noinline} = dense<0.0263157897> : tensor<64xf32>
util.global private @"__iree_flow___sm_node317__m.layer-23.gamma" {noinline} = dense<0.025641026> : tensor<64xf32>
util.global private @"__iree_flow___sm_node318__m.layer-23.beta" {noinline} = dense<2.500000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node319__m.layer-23.moving_mean" {noinline} = dense<0.024390243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node320__m.layer-23.moving_variance" {noinline} = dense<0.0238095243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node329__m.layer-25.kernel" {noinline} = dense<0.0232558139> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node330__m.layer-25.bias" {noinline} = dense<0.0227272734> : tensor<256xf32>
util.global private @"__iree_flow___sm_node336__m.layer-26.gamma" {noinline} = dense<0.0222222228> : tensor<256xf32>
util.global private @"__iree_flow___sm_node337__m.layer-26.beta" {noinline} = dense<0.0217391308> : tensor<256xf32>
util.global private @"__iree_flow___sm_node338__m.layer-26.moving_mean" {noinline} = dense<0.0212765951> : tensor<256xf32>
util.global private @"__iree_flow___sm_node339__m.layer-26.moving_variance" {noinline} = dense<0.020833334> : tensor<256xf32>
util.global private @"__iree_flow___sm_node352__m.layer-29.kernel" {noinline} = dense<0.0204081628> : tensor<1x1x256x64xf32>
util.global private @"__iree_flow___sm_node353__m.layer-29.bias" {noinline} = dense<2.000000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node359__m.layer-30.gamma" {noinline} = dense<0.0196078438> : tensor<64xf32>
util.global private @"__iree_flow___sm_node360__m.layer-30.beta" {noinline} = dense<0.0192307699> : tensor<64xf32>
util.global private @"__iree_flow___sm_node361__m.layer-30.moving_mean" {noinline} = dense<0.0188679248> : tensor<64xf32>
util.global private @"__iree_flow___sm_node362__m.layer-30.moving_variance" {noinline} = dense<0.0185185187> : tensor<64xf32>
util.global private @"__iree_flow___sm_node371__m.layer-32.kernel" {noinline} = dense<0.0181818176> : tensor<3x3x64x64xf32>
util.global private @"__iree_flow___sm_node372__m.layer-32.bias" {noinline} = dense<0.0178571437> : tensor<64xf32>
util.global private @"__iree_flow___sm_node378__m.layer-33.gamma" {noinline} = dense<0.0175438598> : tensor<64xf32>
util.global private @"__iree_flow___sm_node379__m.layer-33.beta" {noinline} = dense<0.0172413792> : tensor<64xf32>
util.global private @"__iree_flow___sm_node380__m.layer-33.moving_mean" {noinline} = dense<0.0169491526> : tensor<64xf32>
util.global private @"__iree_flow___sm_node381__m.layer-33.moving_variance" {noinline} = dense<0.0166666675> : tensor<64xf32>
util.global private @"__iree_flow___sm_node390__m.layer-35.kernel" {noinline} = dense<0.0163934417> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node391__m.layer-35.bias" {noinline} = dense<0.0161290318> : tensor<256xf32>
util.global private @"__iree_flow___sm_node397__m.layer-36.gamma" {noinline} = dense<0.0158730168> : tensor<256xf32>
util.global private @"__iree_flow___sm_node398__m.layer-36.beta" {noinline} = dense<1.562500e-02> : tensor<256xf32>
util.global private @"__iree_flow___sm_node399__m.layer-36.moving_mean" {noinline} = dense<0.0153846154> : tensor<256xf32>
util.global private @"__iree_flow___sm_node400__m.layer-36.moving_variance" {noinline} = dense<0.0151515156> : tensor<256xf32>
util.global private @"__iree_flow___sm_node413__m.layer-39.kernel" {noinline} = dense<0.0149253728> : tensor<1x1x256x128xf32>
util.global private @"__iree_flow___sm_node414__m.layer-39.bias" {noinline} = dense<0.0147058824> : tensor<128xf32>
util.global private @"__iree_flow___sm_node420__m.layer-40.gamma" {noinline} = dense<0.0144927539> : tensor<128xf32>
util.global private @"__iree_flow___sm_node421__m.layer-40.beta" {noinline} = dense<0.0142857144> : tensor<128xf32>
util.global private @"__iree_flow___sm_node422__m.layer-40.moving_mean" {noinline} = dense<0.0140845068> : tensor<128xf32>
util.global private @"__iree_flow___sm_node423__m.layer-40.moving_variance" {noinline} = dense<0.013888889> : tensor<128xf32>
util.global private @"__iree_flow___sm_node432__m.layer-42.kernel" {noinline} = dense<0.01369863> : tensor<3x3x128x128xf32>
util.global private @"__iree_flow___sm_node433__m.layer-42.bias" {noinline} = dense<0.0135135138> : tensor<128xf32>
util.global private @"__iree_flow___sm_node439__m.layer-43.gamma" {noinline} = dense<0.0133333337> : tensor<128xf32>
util.global private @"__iree_flow___sm_node440__m.layer-43.beta" {noinline} = dense<0.0131578948> : tensor<128xf32>
util.global private @"__iree_flow___sm_node441__m.layer-43.moving_mean" {noinline} = dense<0.012987013> : tensor<128xf32>
util.global private @"__iree_flow___sm_node442__m.layer-43.moving_variance" {noinline} = dense<0.012820513> : tensor<128xf32>
util.global private @"__iree_flow___sm_node451__m.layer-45.kernel" {noinline} = dense<0.0126582282> : tensor<1x1x256x512xf32>
util.global private @"__iree_flow___sm_node452__m.layer-45.bias" {noinline} = dense<1.250000e-02> : tensor<512xf32>
util.global private @"__iree_flow___sm_node457__m.layer-46.kernel" {noinline} = dense<0.0123456791> : tensor<1x1x128x512xf32>
util.global private @"__iree_flow___sm_node458__m.layer-46.bias" {noinline} = dense<0.0121951215> : tensor<512xf32>
util.global private @"__iree_flow___sm_node464__m.layer-47.gamma" {noinline} = dense<0.0120481923> : tensor<512xf32>
util.global private @"__iree_flow___sm_node465__m.layer-47.beta" {noinline} = dense<0.0119047621> : tensor<512xf32>
util.global private @"__iree_flow___sm_node466__m.layer-47.moving_mean" {noinline} = dense<0.0117647061> : tensor<512xf32>
util.global private @"__iree_flow___sm_node467__m.layer-47.moving_variance" {noinline} = dense<0.0116279069> : tensor<512xf32>
util.global private @"__iree_flow___sm_node473__m.layer-48.gamma" {noinline} = dense<0.0114942528> : tensor<512xf32>
util.global private @"__iree_flow___sm_node474__m.layer-48.beta" {noinline} = dense<0.0113636367> : tensor<512xf32>
util.global private @"__iree_flow___sm_node475__m.layer-48.moving_mean" {noinline} = dense<0.0112359552> : tensor<512xf32>
util.global private @"__iree_flow___sm_node476__m.layer-48.moving_variance" {noinline} = dense<0.0111111114> : tensor<512xf32>
util.global private @"__iree_flow___sm_node489__m.layer-51.kernel" {noinline} = dense<0.0109890113> : tensor<1x1x512x128xf32>
util.global private @"__iree_flow___sm_node490__m.layer-51.bias" {noinline} = dense<0.0108695654> : tensor<128xf32>
util.global private @"__iree_flow___sm_node496__m.layer-52.gamma" {noinline} = dense<0.0107526882> : tensor<128xf32>
util.global private @"__iree_flow___sm_node497__m.layer-52.beta" {noinline} = dense<0.0106382975> : tensor<128xf32>
util.global private @"__iree_flow___sm_node498__m.layer-52.moving_mean" {noinline} = dense<0.0105263162> : tensor<128xf32>
util.global private @"__iree_flow___sm_node499__m.layer-52.moving_variance" {noinline} = dense<0.010416667> : tensor<128xf32>
util.global private @"__iree_flow___sm_node508__m.layer-54.kernel" {noinline} = dense<0.010309278> : tensor<3x3x128x128xf32>
util.global private @"__iree_flow___sm_node509__m.layer-54.bias" {noinline} = dense<0.0102040814> : tensor<128xf32>
util.global private @"__iree_flow___sm_node515__m.layer-55.gamma" {noinline} = dense<0.0101010101> : tensor<128xf32>
util.global private @"__iree_flow___sm_node516__m.layer-55.beta" {noinline} = dense<0.00999999977> : tensor<128xf32>
util.global private @"__iree_flow___sm_node517__m.layer-55.moving_mean" {noinline} = dense<9.900990e-03> : tensor<128xf32>
util.global private @"__iree_flow___sm_node518__m.layer-55.moving_variance" {noinline} = dense<0.00980392192> : tensor<128xf32>
util.global private @"__iree_flow___sm_node527__m.layer-57.kernel" {noinline} = dense<0.00970873795> : tensor<1x1x128x512xf32>
util.global private @"__iree_flow___sm_node528__m.layer-57.bias" {noinline} = dense<0.00961538497> : tensor<512xf32>
util.global private @"__iree_flow___sm_node534__m.layer-58.gamma" {noinline} = dense<9.523810e-03> : tensor<512xf32>
util.global private @"__iree_flow___sm_node535__m.layer-58.beta" {noinline} = dense<0.0094339624> : tensor<512xf32>
util.global private @"__iree_flow___sm_node536__m.layer-58.moving_mean" {noinline} = dense<0.00934579409> : tensor<512xf32>
util.global private @"__iree_flow___sm_node537__m.layer-58.moving_variance" {noinline} = dense<0.00925925932> : tensor<512xf32>
util.global private @"__iree_flow___sm_node550__m.layer-61.kernel" {noinline} = dense<0.00917431153> : tensor<1x1x512x128xf32>
util.global private @"__iree_flow___sm_node551__m.layer-61.bias" {noinline} = dense<0.0090909088> : tensor<128xf32>
util.global private @"__iree_flow___sm_node557__m.layer-62.gamma" {noinline} = dense<0.00900900922> : tensor<128xf32>
util.global private @"__iree_flow___sm_node558__m.layer-62.beta" {noinline} = dense<0.00892857183> : tensor<128xf32>
util.global private @"__iree_flow___sm_node559__m.layer-62.moving_mean" {noinline} = dense<0.00884955748> : tensor<128xf32>
util.global private @"__iree_flow___sm_node560__m.layer-62.moving_variance" {noinline} = dense<0.00877192988> : tensor<128xf32>
util.global private @"__iree_flow___sm_node569__m.layer-64.kernel" {noinline} = dense<0.00869565178> : tensor<3x3x128x128xf32>
util.global private @"__iree_flow___sm_node570__m.layer-64.bias" {noinline} = dense<8.620690e-03> : tensor<128xf32>
util.global private @"__iree_flow___sm_node576__m.layer-65.gamma" {noinline} = dense<0.00854700897> : tensor<128xf32>
util.global private @"__iree_flow___sm_node577__m.layer-65.beta" {noinline} = dense<0.00847457629> : tensor<128xf32>
util.global private @"__iree_flow___sm_node578__m.layer-65.moving_mean" {noinline} = dense<0.00840336177> : tensor<128xf32>
util.global private @"__iree_flow___sm_node579__m.layer-65.moving_variance" {noinline} = dense<0.00833333377> : tensor<128xf32>
util.global private @"__iree_flow___sm_node588__m.layer-67.kernel" {noinline} = dense<0.00826446246> : tensor<1x1x128x512xf32>
util.global private @"__iree_flow___sm_node589__m.layer-67.bias" {noinline} = dense<0.00819672085> : tensor<512xf32>
util.global private @"__iree_flow___sm_node595__m.layer-68.gamma" {noinline} = dense<0.008130081> : tensor<512xf32>
util.global private @"__iree_flow___sm_node596__m.layer-68.beta" {noinline} = dense<0.00806451589> : tensor<512xf32>
util.global private @"__iree_flow___sm_node597__m.layer-68.moving_mean" {noinline} = dense<8.000000e-03> : tensor<512xf32>
util.global private @"__iree_flow___sm_node598__m.layer-68.moving_variance" {noinline} = dense<0.00793650839> : tensor<512xf32>
util.global private @"__iree_flow___sm_node611__m.layer-71.kernel" {noinline} = dense<0.00787401571> : tensor<1x1x512x128xf32>
util.global private @"__iree_flow___sm_node612__m.layer-71.bias" {noinline} = dense<7.812500e-03> : tensor<128xf32>
util.global private @"__iree_flow___sm_node618__m.layer-72.gamma" {noinline} = dense<0.00775193795> : tensor<128xf32>
util.global private @"__iree_flow___sm_node619__m.layer-72.beta" {noinline} = dense<0.0076923077> : tensor<128xf32>
util.global private @"__iree_flow___sm_node620__m.layer-72.moving_mean" {noinline} = dense<0.00763358781> : tensor<128xf32>
util.global private @"__iree_flow___sm_node621__m.layer-72.moving_variance" {noinline} = dense<0.0075757578> : tensor<128xf32>
util.global private @"__iree_flow___sm_node630__m.layer-74.kernel" {noinline} = dense<0.00751879718> : tensor<3x3x128x128xf32>
util.global private @"__iree_flow___sm_node631__m.layer-74.bias" {noinline} = dense<0.00746268639> : tensor<128xf32>
util.global private @"__iree_flow___sm_node637__m.layer-75.gamma" {noinline} = dense<0.00740740728> : tensor<128xf32>
util.global private @"__iree_flow___sm_node638__m.layer-75.beta" {noinline} = dense<0.0073529412> : tensor<128xf32>
util.global private @"__iree_flow___sm_node639__m.layer-75.moving_mean" {noinline} = dense<7.299270e-03> : tensor<128xf32>
util.global private @"__iree_flow___sm_node640__m.layer-75.moving_variance" {noinline} = dense<0.00724637694> : tensor<128xf32>
util.global private @"__iree_flow___sm_node649__m.layer-77.kernel" {noinline} = dense<0.00719424477> : tensor<1x1x128x512xf32>
util.global private @"__iree_flow___sm_node650__m.layer-77.bias" {noinline} = dense<0.00714285718> : tensor<512xf32>
util.global private @"__iree_flow___sm_node656__m.layer-78.gamma" {noinline} = dense<0.00709219835> : tensor<512xf32>
util.global private @"__iree_flow___sm_node657__m.layer-78.beta" {noinline} = dense<0.00704225338> : tensor<512xf32>
util.global private @"__iree_flow___sm_node658__m.layer-78.moving_mean" {noinline} = dense<0.00699300691> : tensor<512xf32>
util.global private @"__iree_flow___sm_node659__m.layer-78.moving_variance" {noinline} = dense<0.0069444445> : tensor<512xf32>
util.global private @"__iree_flow___sm_node672__m.layer-81.kernel" {noinline} = dense<0.0068965517> : tensor<1x1x512x256xf32>
util.global private @"__iree_flow___sm_node673__m.layer-81.bias" {noinline} = dense<0.00684931502> : tensor<256xf32>
util.global private @"__iree_flow___sm_node679__m.layer-82.gamma" {noinline} = dense<0.00680272094> : tensor<256xf32>
util.global private @"__iree_flow___sm_node680__m.layer-82.beta" {noinline} = dense<0.00675675692> : tensor<256xf32>
util.global private @"__iree_flow___sm_node681__m.layer-82.moving_mean" {noinline} = dense<0.00671140943> : tensor<256xf32>
util.global private @"__iree_flow___sm_node682__m.layer-82.moving_variance" {noinline} = dense<0.00666666683> : tensor<256xf32>
util.global private @"__iree_flow___sm_node691__m.layer-84.kernel" {noinline} = dense<0.00662251655> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node692__m.layer-84.bias" {noinline} = dense<0.00657894742> : tensor<256xf32>
util.global private @"__iree_flow___sm_node698__m.layer-85.gamma" {noinline} = dense<0.00653594779> : tensor<256xf32>
util.global private @"__iree_flow___sm_node699__m.layer-85.beta" {noinline} = dense<0.00649350649> : tensor<256xf32>
util.global private @"__iree_flow___sm_node700__m.layer-85.moving_mean" {noinline} = dense<0.0064516128> : tensor<256xf32>
util.global private @"__iree_flow___sm_node701__m.layer-85.moving_variance" {noinline} = dense<0.00641025649> : tensor<256xf32>
util.global private @"__iree_flow___sm_node710__m.layer-87.kernel" {noinline} = dense<0.00636942684> : tensor<1x1x512x1024xf32>
util.global private @"__iree_flow___sm_node711__m.layer-87.bias" {noinline} = dense<0.00632911408> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node716__m.layer-88.kernel" {noinline} = dense<0.00628930796> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node717__m.layer-88.bias" {noinline} = dense<6.250000e-03> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node723__m.layer-89.gamma" {noinline} = dense<0.00621118024> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node724__m.layer-89.beta" {noinline} = dense<0.00617283955> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node725__m.layer-89.moving_mean" {noinline} = dense<0.00613496918> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node726__m.layer-89.moving_variance" {noinline} = dense<0.00609756075> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node732__m.layer-90.gamma" {noinline} = dense<0.00606060587> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node733__m.layer-90.beta" {noinline} = dense<0.00602409616> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node734__m.layer-90.moving_mean" {noinline} = dense<0.00598802418> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node735__m.layer-90.moving_variance" {noinline} = dense<0.00595238106> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node748__m.layer-93.kernel" {noinline} = dense<5.917160e-03> : tensor<1x1x1024x256xf32>
util.global private @"__iree_flow___sm_node749__m.layer-93.bias" {noinline} = dense<0.00588235306> : tensor<256xf32>
util.global private @"__iree_flow___sm_node755__m.layer-94.gamma" {noinline} = dense<0.00584795326> : tensor<256xf32>
util.global private @"__iree_flow___sm_node756__m.layer-94.beta" {noinline} = dense<0.00581395347> : tensor<256xf32>
util.global private @"__iree_flow___sm_node757__m.layer-94.moving_mean" {noinline} = dense<0.00578034669> : tensor<256xf32>
util.global private @"__iree_flow___sm_node758__m.layer-94.moving_variance" {noinline} = dense<0.00574712642> : tensor<256xf32>
util.global private @"__iree_flow___sm_node767__m.layer-96.kernel" {noinline} = dense<0.00571428565> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node768__m.layer-96.bias" {noinline} = dense<0.00568181835> : tensor<256xf32>
util.global private @"__iree_flow___sm_node774__m.layer-97.gamma" {noinline} = dense<0.00564971752> : tensor<256xf32>
util.global private @"__iree_flow___sm_node775__m.layer-97.beta" {noinline} = dense<0.00561797759> : tensor<256xf32>
util.global private @"__iree_flow___sm_node776__m.layer-97.moving_mean" {noinline} = dense<0.00558659201> : tensor<256xf32>
util.global private @"__iree_flow___sm_node777__m.layer-97.moving_variance" {noinline} = dense<0.00555555569> : tensor<256xf32>
util.global private @"__iree_flow___sm_node786__m.layer-99.kernel" {noinline} = dense<0.00552486209> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node787__m.layer-99.bias" {noinline} = dense<0.00549450563> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node793__m.layer-100.gamma" {noinline} = dense<0.00546448072> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node794__m.layer-100.beta" {noinline} = dense<0.00543478271> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node795__m.layer-100.moving_mean" {noinline} = dense<0.00540540554> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node796__m.layer-100.moving_variance" {noinline} = dense<0.00537634408> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node809__m.layer-103.kernel" {noinline} = dense<0.00534759369> : tensor<1x1x1024x256xf32>
util.global private @"__iree_flow___sm_node810__m.layer-103.bias" {noinline} = dense<0.00531914877> : tensor<256xf32>
util.global private @"__iree_flow___sm_node816__m.layer-104.gamma" {noinline} = dense<0.00529100513> : tensor<256xf32>
util.global private @"__iree_flow___sm_node817__m.layer-104.beta" {noinline} = dense<0.00526315812> : tensor<256xf32>
util.global private @"__iree_flow___sm_node818__m.layer-104.moving_mean" {noinline} = dense<0.00523560215> : tensor<256xf32>
util.global private @"__iree_flow___sm_node819__m.layer-104.moving_variance" {noinline} = dense<0.00520833349> : tensor<256xf32>
util.global private @"__iree_flow___sm_node828__m.layer-106.kernel" {noinline} = dense<0.00518134702> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node829__m.layer-106.bias" {noinline} = dense<0.00515463902> : tensor<256xf32>
util.global private @"__iree_flow___sm_node835__m.layer-107.gamma" {noinline} = dense<0.00512820529> : tensor<256xf32>
util.global private @"__iree_flow___sm_node836__m.layer-107.beta" {noinline} = dense<0.00510204071> : tensor<256xf32>
util.global private @"__iree_flow___sm_node837__m.layer-107.moving_mean" {noinline} = dense<0.00507614203> : tensor<256xf32>
util.global private @"__iree_flow___sm_node838__m.layer-107.moving_variance" {noinline} = dense<0.00505050505> : tensor<256xf32>
util.global private @"__iree_flow___sm_node847__m.layer-109.kernel" {noinline} = dense<0.00502512557> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node848__m.layer-109.bias" {noinline} = dense<5.000000e-03> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node854__m.layer-110.gamma" {noinline} = dense<0.00497512426> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node855__m.layer-110.beta" {noinline} = dense<0.00495049497> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node856__m.layer-110.moving_mean" {noinline} = dense<0.00492610829> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node857__m.layer-110.moving_variance" {noinline} = dense<0.00490196096> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node870__m.layer-113.kernel" {noinline} = dense<0.00487804879> : tensor<1x1x1024x256xf32>
util.global private @"__iree_flow___sm_node871__m.layer-113.bias" {noinline} = dense<0.00485436898> : tensor<256xf32>
util.global private @"__iree_flow___sm_node877__m.layer-114.gamma" {noinline} = dense<0.00483091781> : tensor<256xf32>
util.global private @"__iree_flow___sm_node878__m.layer-114.beta" {noinline} = dense<0.00480769249> : tensor<256xf32>
util.global private @"__iree_flow___sm_node879__m.layer-114.moving_mean" {noinline} = dense<0.00478468882> : tensor<256xf32>
util.global private @"__iree_flow___sm_node880__m.layer-114.moving_variance" {noinline} = dense<0.00476190494> : tensor<256xf32>
util.global private @"__iree_flow___sm_node889__m.layer-116.kernel" {noinline} = dense<0.00473933667> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node890__m.layer-116.bias" {noinline} = dense<0.0047169812> : tensor<256xf32>
util.global private @"__iree_flow___sm_node896__m.layer-117.gamma" {noinline} = dense<0.00469483575> : tensor<256xf32>
util.global private @"__iree_flow___sm_node897__m.layer-117.beta" {noinline} = dense<0.00467289705> : tensor<256xf32>
util.global private @"__iree_flow___sm_node898__m.layer-117.moving_mean" {noinline} = dense<0.00465116277> : tensor<256xf32>
util.global private @"__iree_flow___sm_node899__m.layer-117.moving_variance" {noinline} = dense<0.00462962966> : tensor<256xf32>
util.global private @"__iree_flow___sm_node908__m.layer-119.kernel" {noinline} = dense<0.00460829493> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node909__m.layer-119.bias" {noinline} = dense<0.00458715577> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node915__m.layer-120.gamma" {noinline} = dense<4.566210e-03> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node916__m.layer-120.beta" {noinline} = dense<0.0045454544> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node917__m.layer-120.moving_mean" {noinline} = dense<0.00452488707> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node918__m.layer-120.moving_variance" {noinline} = dense<0.00450450461> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node931__m.layer-123.kernel" {noinline} = dense<0.00448430516> : tensor<1x1x1024x256xf32>
util.global private @"__iree_flow___sm_node932__m.layer-123.bias" {noinline} = dense<0.00446428591> : tensor<256xf32>
util.global private @"__iree_flow___sm_node938__m.layer-124.gamma" {noinline} = dense<0.00444444455> : tensor<256xf32>
util.global private @"__iree_flow___sm_node939__m.layer-124.beta" {noinline} = dense<0.00442477874> : tensor<256xf32>
util.global private @"__iree_flow___sm_node940__m.layer-124.moving_mean" {noinline} = dense<0.00440528616> : tensor<256xf32>
util.global private @"__iree_flow___sm_node941__m.layer-124.moving_variance" {noinline} = dense<0.00438596494> : tensor<256xf32>
util.global private @"__iree_flow___sm_node950__m.layer-126.kernel" {noinline} = dense<0.0043668123> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node951__m.layer-126.bias" {noinline} = dense<0.00434782589> : tensor<256xf32>
util.global private @"__iree_flow___sm_node957__m.layer-127.gamma" {noinline} = dense<0.00432900432> : tensor<256xf32>
util.global private @"__iree_flow___sm_node958__m.layer-127.beta" {noinline} = dense<0.00431034481> : tensor<256xf32>
util.global private @"__iree_flow___sm_node959__m.layer-127.moving_mean" {noinline} = dense<0.00429184549> : tensor<256xf32>
util.global private @"__iree_flow___sm_node960__m.layer-127.moving_variance" {noinline} = dense<0.00427350448> : tensor<256xf32>
util.global private @"__iree_flow___sm_node969__m.layer-129.kernel" {noinline} = dense<0.00425531901> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node970__m.layer-129.bias" {noinline} = dense<0.00423728814> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node976__m.layer-130.gamma" {noinline} = dense<0.00421940908> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node977__m.layer-130.beta" {noinline} = dense<0.00420168089> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node978__m.layer-130.moving_mean" {noinline} = dense<0.00418410031> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node979__m.layer-130.moving_variance" {noinline} = dense<0.00416666688> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node992__m.layer-133.kernel" {noinline} = dense<0.00414937781> : tensor<1x1x1024x256xf32>
util.global private @"__iree_flow___sm_node993__m.layer-133.bias" {noinline} = dense<0.00413223123> : tensor<256xf32>
util.global private @"__iree_flow___sm_node999__m.layer-134.gamma" {noinline} = dense<0.00411522621> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1000__m.layer-134.beta" {noinline} = dense<0.00409836043> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1001__m.layer-134.moving_mean" {noinline} = dense<0.00408163248> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1002__m.layer-134.moving_variance" {noinline} = dense<0.0040650405> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1011__m.layer-136.kernel" {noinline} = dense<0.0040485831> : tensor<3x3x256x256xf32>
util.global private @"__iree_flow___sm_node1012__m.layer-136.bias" {noinline} = dense<0.00403225794> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1018__m.layer-137.gamma" {noinline} = dense<0.00401606411> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1019__m.layer-137.beta" {noinline} = dense<4.000000e-03> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1020__m.layer-137.moving_mean" {noinline} = dense<0.00398406386> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1021__m.layer-137.moving_variance" {noinline} = dense<0.0039682542> : tensor<256xf32>
util.global private @"__iree_flow___sm_node1030__m.layer-139.kernel" {noinline} = dense<0.00395256933> : tensor<1x1x256x1024xf32>
util.global private @"__iree_flow___sm_node1031__m.layer-139.bias" {noinline} = dense<0.00393700786> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node1037__m.layer-140.gamma" {noinline} = dense<0.00392156886> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node1038__m.layer-140.beta" {noinline} = dense<3.906250e-03> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node1039__m.layer-140.moving_mean" {noinline} = dense<0.00389105058> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node1040__m.layer-140.moving_variance" {noinline} = dense<0.00387596898> : tensor<1024xf32>
util.global private @"__iree_flow___sm_node1053__m.layer-143.kernel" {noinline} = dense<0.00386100379> : tensor<1x1x1024x512xf32>
util.global private @"__iree_flow___sm_node1054__m.layer-143.bias" {noinline} = dense<0.00384615385> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1060__m.layer-144.gamma" {noinline} = dense<0.00383141753> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1061__m.layer-144.beta" {noinline} = dense<0.00381679391> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1062__m.layer-144.moving_mean" {noinline} = dense<0.00380228134> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1063__m.layer-144.moving_variance" {noinline} = dense<0.0037878789> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1072__m.layer-146.kernel" {noinline} = dense<0.00377358496> : tensor<3x3x512x512xf32>
util.global private @"__iree_flow___sm_node1073__m.layer-146.bias" {noinline} = dense<0.00375939859> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1079__m.layer-147.gamma" {noinline} = dense<0.00374531839> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1080__m.layer-147.beta" {noinline} = dense<0.0037313432> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1081__m.layer-147.moving_mean" {noinline} = dense<0.00371747208> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1082__m.layer-147.moving_variance" {noinline} = dense<0.00370370364> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1091__m.layer-149.kernel" {noinline} = dense<0.00369003695> : tensor<1x1x1024x2048xf32>
util.global private @"__iree_flow___sm_node1092__m.layer-149.bias" {noinline} = dense<0.0036764706> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1097__m.layer-150.kernel" {noinline} = dense<0.00366300368> : tensor<1x1x512x2048xf32>
util.global private @"__iree_flow___sm_node1098__m.layer-150.bias" {noinline} = dense<0.00364963501> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1104__m.layer-151.gamma" {noinline} = dense<0.00363636366> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1105__m.layer-151.beta" {noinline} = dense<0.00362318847> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1106__m.layer-151.moving_mean" {noinline} = dense<0.00361010828> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1107__m.layer-151.moving_variance" {noinline} = dense<0.00359712238> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1113__m.layer-152.gamma" {noinline} = dense<0.00358422939> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1114__m.layer-152.beta" {noinline} = dense<0.00357142859> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1115__m.layer-152.moving_mean" {noinline} = dense<0.00355871883> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1116__m.layer-152.moving_variance" {noinline} = dense<0.00354609918> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1129__m.layer-155.kernel" {noinline} = dense<0.00353356893> : tensor<1x1x2048x512xf32>
util.global private @"__iree_flow___sm_node1130__m.layer-155.bias" {noinline} = dense<0.00352112669> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1136__m.layer-156.gamma" {noinline} = dense<0.003508772> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1137__m.layer-156.beta" {noinline} = dense<0.00349650346> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1138__m.layer-156.moving_mean" {noinline} = dense<0.00348432059> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1139__m.layer-156.moving_variance" {noinline} = dense<0.00347222225> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1148__m.layer-158.kernel" {noinline} = dense<0.00346020772> : tensor<3x3x512x512xf32>
util.global private @"__iree_flow___sm_node1149__m.layer-158.bias" {noinline} = dense<0.00344827585> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1155__m.layer-159.gamma" {noinline} = dense<0.00343642617> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1156__m.layer-159.beta" {noinline} = dense<0.00342465751> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1157__m.layer-159.moving_mean" {noinline} = dense<0.00341296918> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1158__m.layer-159.moving_variance" {noinline} = dense<0.00340136047> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1167__m.layer-161.kernel" {noinline} = dense<0.00338983047> : tensor<1x1x512x2048xf32>
util.global private @"__iree_flow___sm_node1168__m.layer-161.bias" {noinline} = dense<0.00337837846> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1174__m.layer-162.gamma" {noinline} = dense<0.00336700329> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1175__m.layer-162.beta" {noinline} = dense<0.00335570471> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1176__m.layer-162.moving_mean" {noinline} = dense<0.00334448158> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1177__m.layer-162.moving_variance" {noinline} = dense<0.00333333341> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1190__m.layer-165.kernel" {noinline} = dense<0.00332225906> : tensor<1x1x2048x512xf32>
util.global private @"__iree_flow___sm_node1191__m.layer-165.bias" {noinline} = dense<0.00331125828> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1197__m.layer-166.gamma" {noinline} = dense<0.00330033014> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1198__m.layer-166.beta" {noinline} = dense<0.00328947371> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1199__m.layer-166.moving_mean" {noinline} = dense<0.00327868853> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1200__m.layer-166.moving_variance" {noinline} = dense<0.00326797389> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1209__m.layer-168.kernel" {noinline} = dense<0.00325732888> : tensor<3x3x512x512xf32>
util.global private @"__iree_flow___sm_node1210__m.layer-168.bias" {noinline} = dense<0.00324675324> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1216__m.layer-169.gamma" {noinline} = dense<0.00323624606> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1217__m.layer-169.beta" {noinline} = dense<0.0032258064> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1218__m.layer-169.moving_mean" {noinline} = dense<0.00321543403> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1219__m.layer-169.moving_variance" {noinline} = dense<0.00320512825> : tensor<512xf32>
util.global private @"__iree_flow___sm_node1228__m.layer-171.kernel" {noinline} = dense<0.00319488812> : tensor<1x1x512x2048xf32>
util.global private @"__iree_flow___sm_node1229__m.layer-171.bias" {noinline} = dense<0.00318471342> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1235__m.layer-172.gamma" {noinline} = dense<0.00317460322> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1236__m.layer-172.beta" {noinline} = dense<0.00316455704> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1237__m.layer-172.moving_mean" {noinline} = dense<0.00315457419> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1238__m.layer-172.moving_variance" {noinline} = dense<0.00314465398> : tensor<2048xf32>
util.global private @"__iree_flow___sm_node1255__m.layer-176.kernel" {noinline} = dense<0.00313479616> : tensor<2048x1000xf32>
util.global private @"__iree_flow___sm_node1256__m.layer-176.bias" {noinline} = dense<3.125000e-03> : tensor<1000xf32>

func.func @predict(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}} {
  %0 = util.global.address @"__iree_flow___sm_node188__m.layer-2.kernel" : !util.ptr<tensor<7x7x3x64xf32>>
  %1 = util.global.address @"__iree_flow___sm_node189__m.layer-2.bias" : !util.ptr<tensor<64xf32>>
  %2 = util.global.address @"__iree_flow___sm_node195__m.layer-3.gamma" : !util.ptr<tensor<64xf32>>
  %3 = util.global.address @"__iree_flow___sm_node196__m.layer-3.beta" : !util.ptr<tensor<64xf32>>
  %4 = util.global.address @"__iree_flow___sm_node197__m.layer-3.moving_mean" : !util.ptr<tensor<64xf32>>
  %5 = util.global.address @"__iree_flow___sm_node198__m.layer-3.moving_variance" : !util.ptr<tensor<64xf32>>
  %6 = util.global.address @"__iree_flow___sm_node215__m.layer-7.kernel" : !util.ptr<tensor<1x1x64x64xf32>>
  %7 = util.global.address @"__iree_flow___sm_node216__m.layer-7.bias" : !util.ptr<tensor<64xf32>>
  %8 = util.global.address @"__iree_flow___sm_node222__m.layer-8.gamma" : !util.ptr<tensor<64xf32>>
  %9 = util.global.address @"__iree_flow___sm_node223__m.layer-8.beta" : !util.ptr<tensor<64xf32>>
  %10 = util.global.address @"__iree_flow___sm_node224__m.layer-8.moving_mean" : !util.ptr<tensor<64xf32>>
  %11 = util.global.address @"__iree_flow___sm_node225__m.layer-8.moving_variance" : !util.ptr<tensor<64xf32>>
  %12 = util.global.address @"__iree_flow___sm_node234__m.layer-10.kernel" : !util.ptr<tensor<3x3x64x64xf32>>
  %13 = util.global.address @"__iree_flow___sm_node235__m.layer-10.bias" : !util.ptr<tensor<64xf32>>
  %14 = util.global.address @"__iree_flow___sm_node241__m.layer-11.gamma" : !util.ptr<tensor<64xf32>>
  %15 = util.global.address @"__iree_flow___sm_node242__m.layer-11.beta" : !util.ptr<tensor<64xf32>>
  %16 = util.global.address @"__iree_flow___sm_node243__m.layer-11.moving_mean" : !util.ptr<tensor<64xf32>>
  %17 = util.global.address @"__iree_flow___sm_node244__m.layer-11.moving_variance" : !util.ptr<tensor<64xf32>>
  %18 = util.global.address @"__iree_flow___sm_node259__m.layer-14.kernel" : !util.ptr<tensor<1x1x64x256xf32>>
  %19 = util.global.address @"__iree_flow___sm_node260__m.layer-14.bias" : !util.ptr<tensor<256xf32>>
  %20 = util.global.address @"__iree_flow___sm_node253__m.layer-13.kernel" : !util.ptr<tensor<1x1x64x256xf32>>
  %21 = util.global.address @"__iree_flow___sm_node254__m.layer-13.bias" : !util.ptr<tensor<256xf32>>
  %22 = util.global.address @"__iree_flow___sm_node266__m.layer-15.gamma" : !util.ptr<tensor<256xf32>>
  %23 = util.global.address @"__iree_flow___sm_node267__m.layer-15.beta" : !util.ptr<tensor<256xf32>>
  %24 = util.global.address @"__iree_flow___sm_node268__m.layer-15.moving_mean" : !util.ptr<tensor<256xf32>>
  %25 = util.global.address @"__iree_flow___sm_node269__m.layer-15.moving_variance" : !util.ptr<tensor<256xf32>>
  %26 = util.global.address @"__iree_flow___sm_node275__m.layer-16.gamma" : !util.ptr<tensor<256xf32>>
  %27 = util.global.address @"__iree_flow___sm_node276__m.layer-16.beta" : !util.ptr<tensor<256xf32>>
  %28 = util.global.address @"__iree_flow___sm_node277__m.layer-16.moving_mean" : !util.ptr<tensor<256xf32>>
  %29 = util.global.address @"__iree_flow___sm_node278__m.layer-16.moving_variance" : !util.ptr<tensor<256xf32>>
  %30 = util.global.address @"__iree_flow___sm_node291__m.layer-19.kernel" : !util.ptr<tensor<1x1x256x64xf32>>
  %31 = util.global.address @"__iree_flow___sm_node292__m.layer-19.bias" : !util.ptr<tensor<64xf32>>
  %32 = util.global.address @"__iree_flow___sm_node298__m.layer-20.gamma" : !util.ptr<tensor<64xf32>>
  %33 = util.global.address @"__iree_flow___sm_node299__m.layer-20.beta" : !util.ptr<tensor<64xf32>>
  %34 = util.global.address @"__iree_flow___sm_node300__m.layer-20.moving_mean" : !util.ptr<tensor<64xf32>>
  %35 = util.global.address @"__iree_flow___sm_node301__m.layer-20.moving_variance" : !util.ptr<tensor<64xf32>>
  %36 = util.global.address @"__iree_flow___sm_node310__m.layer-22.kernel" : !util.ptr<tensor<3x3x64x64xf32>>
  %37 = util.global.address @"__iree_flow___sm_node311__m.layer-22.bias" : !util.ptr<tensor<64xf32>>
  %38 = util.global.address @"__iree_flow___sm_node317__m.layer-23.gamma" : !util.ptr<tensor<64xf32>>
  %39 = util.global.address @"__iree_flow___sm_node318__m.layer-23.beta" : !util.ptr<tensor<64xf32>>
  %40 = util.global.address @"__iree_flow___sm_node319__m.layer-23.moving_mean" : !util.ptr<tensor<64xf32>>
  %41 = util.global.address @"__iree_flow___sm_node320__m.layer-23.moving_variance" : !util.ptr<tensor<64xf32>>
  %42 = util.global.address @"__iree_flow___sm_node329__m.layer-25.kernel" : !util.ptr<tensor<1x1x64x256xf32>>
  %43 = util.global.address @"__iree_flow___sm_node330__m.layer-25.bias" : !util.ptr<tensor<256xf32>>
  %44 = util.global.address @"__iree_flow___sm_node336__m.layer-26.gamma" : !util.ptr<tensor<256xf32>>
  %45 = util.global.address @"__iree_flow___sm_node337__m.layer-26.beta" : !util.ptr<tensor<256xf32>>
  %46 = util.global.address @"__iree_flow___sm_node338__m.layer-26.moving_mean" : !util.ptr<tensor<256xf32>>
  %47 = util.global.address @"__iree_flow___sm_node339__m.layer-26.moving_variance" : !util.ptr<tensor<256xf32>>
  %48 = util.global.address @"__iree_flow___sm_node352__m.layer-29.kernel" : !util.ptr<tensor<1x1x256x64xf32>>
  %49 = util.global.address @"__iree_flow___sm_node353__m.layer-29.bias" : !util.ptr<tensor<64xf32>>
  %50 = util.global.address @"__iree_flow___sm_node359__m.layer-30.gamma" : !util.ptr<tensor<64xf32>>
  %51 = util.global.address @"__iree_flow___sm_node360__m.layer-30.beta" : !util.ptr<tensor<64xf32>>
  %52 = util.global.address @"__iree_flow___sm_node361__m.layer-30.moving_mean" : !util.ptr<tensor<64xf32>>
  %53 = util.global.address @"__iree_flow___sm_node362__m.layer-30.moving_variance" : !util.ptr<tensor<64xf32>>
  %54 = util.global.address @"__iree_flow___sm_node371__m.layer-32.kernel" : !util.ptr<tensor<3x3x64x64xf32>>
  %55 = util.global.address @"__iree_flow___sm_node372__m.layer-32.bias" : !util.ptr<tensor<64xf32>>
  %56 = util.global.address @"__iree_flow___sm_node378__m.layer-33.gamma" : !util.ptr<tensor<64xf32>>
  %57 = util.global.address @"__iree_flow___sm_node379__m.layer-33.beta" : !util.ptr<tensor<64xf32>>
  %58 = util.global.address @"__iree_flow___sm_node380__m.layer-33.moving_mean" : !util.ptr<tensor<64xf32>>
  %59 = util.global.address @"__iree_flow___sm_node381__m.layer-33.moving_variance" : !util.ptr<tensor<64xf32>>
  %60 = util.global.address @"__iree_flow___sm_node390__m.layer-35.kernel" : !util.ptr<tensor<1x1x64x256xf32>>
  %61 = util.global.address @"__iree_flow___sm_node391__m.layer-35.bias" : !util.ptr<tensor<256xf32>>
  %62 = util.global.address @"__iree_flow___sm_node397__m.layer-36.gamma" : !util.ptr<tensor<256xf32>>
  %63 = util.global.address @"__iree_flow___sm_node398__m.layer-36.beta" : !util.ptr<tensor<256xf32>>
  %64 = util.global.address @"__iree_flow___sm_node399__m.layer-36.moving_mean" : !util.ptr<tensor<256xf32>>
  %65 = util.global.address @"__iree_flow___sm_node400__m.layer-36.moving_variance" : !util.ptr<tensor<256xf32>>
  %66 = util.global.address @"__iree_flow___sm_node413__m.layer-39.kernel" : !util.ptr<tensor<1x1x256x128xf32>>
  %67 = util.global.address @"__iree_flow___sm_node414__m.layer-39.bias" : !util.ptr<tensor<128xf32>>
  %68 = util.global.address @"__iree_flow___sm_node420__m.layer-40.gamma" : !util.ptr<tensor<128xf32>>
  %69 = util.global.address @"__iree_flow___sm_node421__m.layer-40.beta" : !util.ptr<tensor<128xf32>>
  %70 = util.global.address @"__iree_flow___sm_node422__m.layer-40.moving_mean" : !util.ptr<tensor<128xf32>>
  %71 = util.global.address @"__iree_flow___sm_node423__m.layer-40.moving_variance" : !util.ptr<tensor<128xf32>>
  %72 = util.global.address @"__iree_flow___sm_node432__m.layer-42.kernel" : !util.ptr<tensor<3x3x128x128xf32>>
  %73 = util.global.address @"__iree_flow___sm_node433__m.layer-42.bias" : !util.ptr<tensor<128xf32>>
  %74 = util.global.address @"__iree_flow___sm_node439__m.layer-43.gamma" : !util.ptr<tensor<128xf32>>
  %75 = util.global.address @"__iree_flow___sm_node440__m.layer-43.beta" : !util.ptr<tensor<128xf32>>
  %76 = util.global.address @"__iree_flow___sm_node441__m.layer-43.moving_mean" : !util.ptr<tensor<128xf32>>
  %77 = util.global.address @"__iree_flow___sm_node442__m.layer-43.moving_variance" : !util.ptr<tensor<128xf32>>
  %78 = util.global.address @"__iree_flow___sm_node457__m.layer-46.kernel" : !util.ptr<tensor<1x1x128x512xf32>>
  %79 = util.global.address @"__iree_flow___sm_node458__m.layer-46.bias" : !util.ptr<tensor<512xf32>>
  %80 = util.global.address @"__iree_flow___sm_node451__m.layer-45.kernel" : !util.ptr<tensor<1x1x256x512xf32>>
  %81 = util.global.address @"__iree_flow___sm_node452__m.layer-45.bias" : !util.ptr<tensor<512xf32>>
  %82 = util.global.address @"__iree_flow___sm_node464__m.layer-47.gamma" : !util.ptr<tensor<512xf32>>
  %83 = util.global.address @"__iree_flow___sm_node465__m.layer-47.beta" : !util.ptr<tensor<512xf32>>
  %84 = util.global.address @"__iree_flow___sm_node466__m.layer-47.moving_mean" : !util.ptr<tensor<512xf32>>
  %85 = util.global.address @"__iree_flow___sm_node467__m.layer-47.moving_variance" : !util.ptr<tensor<512xf32>>
  %86 = util.global.address @"__iree_flow___sm_node473__m.layer-48.gamma" : !util.ptr<tensor<512xf32>>
  %87 = util.global.address @"__iree_flow___sm_node474__m.layer-48.beta" : !util.ptr<tensor<512xf32>>
  %88 = util.global.address @"__iree_flow___sm_node475__m.layer-48.moving_mean" : !util.ptr<tensor<512xf32>>
  %89 = util.global.address @"__iree_flow___sm_node476__m.layer-48.moving_variance" : !util.ptr<tensor<512xf32>>
  %90 = util.global.address @"__iree_flow___sm_node489__m.layer-51.kernel" : !util.ptr<tensor<1x1x512x128xf32>>
  %91 = util.global.address @"__iree_flow___sm_node490__m.layer-51.bias" : !util.ptr<tensor<128xf32>>
  %92 = util.global.address @"__iree_flow___sm_node496__m.layer-52.gamma" : !util.ptr<tensor<128xf32>>
  %93 = util.global.address @"__iree_flow___sm_node497__m.layer-52.beta" : !util.ptr<tensor<128xf32>>
  %94 = util.global.address @"__iree_flow___sm_node498__m.layer-52.moving_mean" : !util.ptr<tensor<128xf32>>
  %95 = util.global.address @"__iree_flow___sm_node499__m.layer-52.moving_variance" : !util.ptr<tensor<128xf32>>
  %96 = util.global.address @"__iree_flow___sm_node508__m.layer-54.kernel" : !util.ptr<tensor<3x3x128x128xf32>>
  %97 = util.global.address @"__iree_flow___sm_node509__m.layer-54.bias" : !util.ptr<tensor<128xf32>>
  %98 = util.global.address @"__iree_flow___sm_node515__m.layer-55.gamma" : !util.ptr<tensor<128xf32>>
  %99 = util.global.address @"__iree_flow___sm_node516__m.layer-55.beta" : !util.ptr<tensor<128xf32>>
  %100 = util.global.address @"__iree_flow___sm_node517__m.layer-55.moving_mean" : !util.ptr<tensor<128xf32>>
  %101 = util.global.address @"__iree_flow___sm_node518__m.layer-55.moving_variance" : !util.ptr<tensor<128xf32>>
  %102 = util.global.address @"__iree_flow___sm_node527__m.layer-57.kernel" : !util.ptr<tensor<1x1x128x512xf32>>
  %103 = util.global.address @"__iree_flow___sm_node528__m.layer-57.bias" : !util.ptr<tensor<512xf32>>
  %104 = util.global.address @"__iree_flow___sm_node534__m.layer-58.gamma" : !util.ptr<tensor<512xf32>>
  %105 = util.global.address @"__iree_flow___sm_node535__m.layer-58.beta" : !util.ptr<tensor<512xf32>>
  %106 = util.global.address @"__iree_flow___sm_node536__m.layer-58.moving_mean" : !util.ptr<tensor<512xf32>>
  %107 = util.global.address @"__iree_flow___sm_node537__m.layer-58.moving_variance" : !util.ptr<tensor<512xf32>>
  %108 = util.global.address @"__iree_flow___sm_node550__m.layer-61.kernel" : !util.ptr<tensor<1x1x512x128xf32>>
  %109 = util.global.address @"__iree_flow___sm_node551__m.layer-61.bias" : !util.ptr<tensor<128xf32>>
  %110 = util.global.address @"__iree_flow___sm_node557__m.layer-62.gamma" : !util.ptr<tensor<128xf32>>
  %111 = util.global.address @"__iree_flow___sm_node558__m.layer-62.beta" : !util.ptr<tensor<128xf32>>
  %112 = util.global.address @"__iree_flow___sm_node559__m.layer-62.moving_mean" : !util.ptr<tensor<128xf32>>
  %113 = util.global.address @"__iree_flow___sm_node560__m.layer-62.moving_variance" : !util.ptr<tensor<128xf32>>
  %114 = util.global.address @"__iree_flow___sm_node569__m.layer-64.kernel" : !util.ptr<tensor<3x3x128x128xf32>>
  %115 = util.global.address @"__iree_flow___sm_node570__m.layer-64.bias" : !util.ptr<tensor<128xf32>>
  %116 = util.global.address @"__iree_flow___sm_node576__m.layer-65.gamma" : !util.ptr<tensor<128xf32>>
  %117 = util.global.address @"__iree_flow___sm_node577__m.layer-65.beta" : !util.ptr<tensor<128xf32>>
  %118 = util.global.address @"__iree_flow___sm_node578__m.layer-65.moving_mean" : !util.ptr<tensor<128xf32>>
  %119 = util.global.address @"__iree_flow___sm_node579__m.layer-65.moving_variance" : !util.ptr<tensor<128xf32>>
  %120 = util.global.address @"__iree_flow___sm_node588__m.layer-67.kernel" : !util.ptr<tensor<1x1x128x512xf32>>
  %121 = util.global.address @"__iree_flow___sm_node589__m.layer-67.bias" : !util.ptr<tensor<512xf32>>
  %122 = util.global.address @"__iree_flow___sm_node595__m.layer-68.gamma" : !util.ptr<tensor<512xf32>>
  %123 = util.global.address @"__iree_flow___sm_node596__m.layer-68.beta" : !util.ptr<tensor<512xf32>>
  %124 = util.global.address @"__iree_flow___sm_node597__m.layer-68.moving_mean" : !util.ptr<tensor<512xf32>>
  %125 = util.global.address @"__iree_flow___sm_node598__m.layer-68.moving_variance" : !util.ptr<tensor<512xf32>>
  %126 = util.global.address @"__iree_flow___sm_node611__m.layer-71.kernel" : !util.ptr<tensor<1x1x512x128xf32>>
  %127 = util.global.address @"__iree_flow___sm_node612__m.layer-71.bias" : !util.ptr<tensor<128xf32>>
  %128 = util.global.address @"__iree_flow___sm_node618__m.layer-72.gamma" : !util.ptr<tensor<128xf32>>
  %129 = util.global.address @"__iree_flow___sm_node619__m.layer-72.beta" : !util.ptr<tensor<128xf32>>
  %130 = util.global.address @"__iree_flow___sm_node620__m.layer-72.moving_mean" : !util.ptr<tensor<128xf32>>
  %131 = util.global.address @"__iree_flow___sm_node621__m.layer-72.moving_variance" : !util.ptr<tensor<128xf32>>
  %132 = util.global.address @"__iree_flow___sm_node630__m.layer-74.kernel" : !util.ptr<tensor<3x3x128x128xf32>>
  %133 = util.global.address @"__iree_flow___sm_node631__m.layer-74.bias" : !util.ptr<tensor<128xf32>>
  %134 = util.global.address @"__iree_flow___sm_node637__m.layer-75.gamma" : !util.ptr<tensor<128xf32>>
  %135 = util.global.address @"__iree_flow___sm_node638__m.layer-75.beta" : !util.ptr<tensor<128xf32>>
  %136 = util.global.address @"__iree_flow___sm_node639__m.layer-75.moving_mean" : !util.ptr<tensor<128xf32>>
  %137 = util.global.address @"__iree_flow___sm_node640__m.layer-75.moving_variance" : !util.ptr<tensor<128xf32>>
  %138 = util.global.address @"__iree_flow___sm_node649__m.layer-77.kernel" : !util.ptr<tensor<1x1x128x512xf32>>
  %139 = util.global.address @"__iree_flow___sm_node650__m.layer-77.bias" : !util.ptr<tensor<512xf32>>
  %140 = util.global.address @"__iree_flow___sm_node656__m.layer-78.gamma" : !util.ptr<tensor<512xf32>>
  %141 = util.global.address @"__iree_flow___sm_node657__m.layer-78.beta" : !util.ptr<tensor<512xf32>>
  %142 = util.global.address @"__iree_flow___sm_node658__m.layer-78.moving_mean" : !util.ptr<tensor<512xf32>>
  %143 = util.global.address @"__iree_flow___sm_node659__m.layer-78.moving_variance" : !util.ptr<tensor<512xf32>>
  %144 = util.global.address @"__iree_flow___sm_node672__m.layer-81.kernel" : !util.ptr<tensor<1x1x512x256xf32>>
  %145 = util.global.address @"__iree_flow___sm_node673__m.layer-81.bias" : !util.ptr<tensor<256xf32>>
  %146 = util.global.address @"__iree_flow___sm_node679__m.layer-82.gamma" : !util.ptr<tensor<256xf32>>
  %147 = util.global.address @"__iree_flow___sm_node680__m.layer-82.beta" : !util.ptr<tensor<256xf32>>
  %148 = util.global.address @"__iree_flow___sm_node681__m.layer-82.moving_mean" : !util.ptr<tensor<256xf32>>
  %149 = util.global.address @"__iree_flow___sm_node682__m.layer-82.moving_variance" : !util.ptr<tensor<256xf32>>
  %150 = util.global.address @"__iree_flow___sm_node691__m.layer-84.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %151 = util.global.address @"__iree_flow___sm_node692__m.layer-84.bias" : !util.ptr<tensor<256xf32>>
  %152 = util.global.address @"__iree_flow___sm_node698__m.layer-85.gamma" : !util.ptr<tensor<256xf32>>
  %153 = util.global.address @"__iree_flow___sm_node699__m.layer-85.beta" : !util.ptr<tensor<256xf32>>
  %154 = util.global.address @"__iree_flow___sm_node700__m.layer-85.moving_mean" : !util.ptr<tensor<256xf32>>
  %155 = util.global.address @"__iree_flow___sm_node701__m.layer-85.moving_variance" : !util.ptr<tensor<256xf32>>
  %156 = util.global.address @"__iree_flow___sm_node716__m.layer-88.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %157 = util.global.address @"__iree_flow___sm_node717__m.layer-88.bias" : !util.ptr<tensor<1024xf32>>
  %158 = util.global.address @"__iree_flow___sm_node710__m.layer-87.kernel" : !util.ptr<tensor<1x1x512x1024xf32>>
  %159 = util.global.address @"__iree_flow___sm_node711__m.layer-87.bias" : !util.ptr<tensor<1024xf32>>
  %160 = util.global.address @"__iree_flow___sm_node723__m.layer-89.gamma" : !util.ptr<tensor<1024xf32>>
  %161 = util.global.address @"__iree_flow___sm_node724__m.layer-89.beta" : !util.ptr<tensor<1024xf32>>
  %162 = util.global.address @"__iree_flow___sm_node725__m.layer-89.moving_mean" : !util.ptr<tensor<1024xf32>>
  %163 = util.global.address @"__iree_flow___sm_node726__m.layer-89.moving_variance" : !util.ptr<tensor<1024xf32>>
  %164 = util.global.address @"__iree_flow___sm_node732__m.layer-90.gamma" : !util.ptr<tensor<1024xf32>>
  %165 = util.global.address @"__iree_flow___sm_node733__m.layer-90.beta" : !util.ptr<tensor<1024xf32>>
  %166 = util.global.address @"__iree_flow___sm_node734__m.layer-90.moving_mean" : !util.ptr<tensor<1024xf32>>
  %167 = util.global.address @"__iree_flow___sm_node735__m.layer-90.moving_variance" : !util.ptr<tensor<1024xf32>>
  %168 = util.global.address @"__iree_flow___sm_node748__m.layer-93.kernel" : !util.ptr<tensor<1x1x1024x256xf32>>
  %169 = util.global.address @"__iree_flow___sm_node749__m.layer-93.bias" : !util.ptr<tensor<256xf32>>
  %170 = util.global.address @"__iree_flow___sm_node755__m.layer-94.gamma" : !util.ptr<tensor<256xf32>>
  %171 = util.global.address @"__iree_flow___sm_node756__m.layer-94.beta" : !util.ptr<tensor<256xf32>>
  %172 = util.global.address @"__iree_flow___sm_node757__m.layer-94.moving_mean" : !util.ptr<tensor<256xf32>>
  %173 = util.global.address @"__iree_flow___sm_node758__m.layer-94.moving_variance" : !util.ptr<tensor<256xf32>>
  %174 = util.global.address @"__iree_flow___sm_node767__m.layer-96.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %175 = util.global.address @"__iree_flow___sm_node768__m.layer-96.bias" : !util.ptr<tensor<256xf32>>
  %176 = util.global.address @"__iree_flow___sm_node774__m.layer-97.gamma" : !util.ptr<tensor<256xf32>>
  %177 = util.global.address @"__iree_flow___sm_node775__m.layer-97.beta" : !util.ptr<tensor<256xf32>>
  %178 = util.global.address @"__iree_flow___sm_node776__m.layer-97.moving_mean" : !util.ptr<tensor<256xf32>>
  %179 = util.global.address @"__iree_flow___sm_node777__m.layer-97.moving_variance" : !util.ptr<tensor<256xf32>>
  %180 = util.global.address @"__iree_flow___sm_node786__m.layer-99.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %181 = util.global.address @"__iree_flow___sm_node787__m.layer-99.bias" : !util.ptr<tensor<1024xf32>>
  %182 = util.global.address @"__iree_flow___sm_node793__m.layer-100.gamma" : !util.ptr<tensor<1024xf32>>
  %183 = util.global.address @"__iree_flow___sm_node794__m.layer-100.beta" : !util.ptr<tensor<1024xf32>>
  %184 = util.global.address @"__iree_flow___sm_node795__m.layer-100.moving_mean" : !util.ptr<tensor<1024xf32>>
  %185 = util.global.address @"__iree_flow___sm_node796__m.layer-100.moving_variance" : !util.ptr<tensor<1024xf32>>
  %186 = util.global.address @"__iree_flow___sm_node809__m.layer-103.kernel" : !util.ptr<tensor<1x1x1024x256xf32>>
  %187 = util.global.address @"__iree_flow___sm_node810__m.layer-103.bias" : !util.ptr<tensor<256xf32>>
  %188 = util.global.address @"__iree_flow___sm_node816__m.layer-104.gamma" : !util.ptr<tensor<256xf32>>
  %189 = util.global.address @"__iree_flow___sm_node817__m.layer-104.beta" : !util.ptr<tensor<256xf32>>
  %190 = util.global.address @"__iree_flow___sm_node818__m.layer-104.moving_mean" : !util.ptr<tensor<256xf32>>
  %191 = util.global.address @"__iree_flow___sm_node819__m.layer-104.moving_variance" : !util.ptr<tensor<256xf32>>
  %192 = util.global.address @"__iree_flow___sm_node828__m.layer-106.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %193 = util.global.address @"__iree_flow___sm_node829__m.layer-106.bias" : !util.ptr<tensor<256xf32>>
  %194 = util.global.address @"__iree_flow___sm_node835__m.layer-107.gamma" : !util.ptr<tensor<256xf32>>
  %195 = util.global.address @"__iree_flow___sm_node836__m.layer-107.beta" : !util.ptr<tensor<256xf32>>
  %196 = util.global.address @"__iree_flow___sm_node837__m.layer-107.moving_mean" : !util.ptr<tensor<256xf32>>
  %197 = util.global.address @"__iree_flow___sm_node838__m.layer-107.moving_variance" : !util.ptr<tensor<256xf32>>
  %198 = util.global.address @"__iree_flow___sm_node847__m.layer-109.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %199 = util.global.address @"__iree_flow___sm_node848__m.layer-109.bias" : !util.ptr<tensor<1024xf32>>
  %200 = util.global.address @"__iree_flow___sm_node854__m.layer-110.gamma" : !util.ptr<tensor<1024xf32>>
  %201 = util.global.address @"__iree_flow___sm_node855__m.layer-110.beta" : !util.ptr<tensor<1024xf32>>
  %202 = util.global.address @"__iree_flow___sm_node856__m.layer-110.moving_mean" : !util.ptr<tensor<1024xf32>>
  %203 = util.global.address @"__iree_flow___sm_node857__m.layer-110.moving_variance" : !util.ptr<tensor<1024xf32>>
  %204 = util.global.address @"__iree_flow___sm_node870__m.layer-113.kernel" : !util.ptr<tensor<1x1x1024x256xf32>>
  %205 = util.global.address @"__iree_flow___sm_node871__m.layer-113.bias" : !util.ptr<tensor<256xf32>>
  %206 = util.global.address @"__iree_flow___sm_node877__m.layer-114.gamma" : !util.ptr<tensor<256xf32>>
  %207 = util.global.address @"__iree_flow___sm_node878__m.layer-114.beta" : !util.ptr<tensor<256xf32>>
  %208 = util.global.address @"__iree_flow___sm_node879__m.layer-114.moving_mean" : !util.ptr<tensor<256xf32>>
  %209 = util.global.address @"__iree_flow___sm_node880__m.layer-114.moving_variance" : !util.ptr<tensor<256xf32>>
  %210 = util.global.address @"__iree_flow___sm_node889__m.layer-116.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %211 = util.global.address @"__iree_flow___sm_node890__m.layer-116.bias" : !util.ptr<tensor<256xf32>>
  %212 = util.global.address @"__iree_flow___sm_node896__m.layer-117.gamma" : !util.ptr<tensor<256xf32>>
  %213 = util.global.address @"__iree_flow___sm_node897__m.layer-117.beta" : !util.ptr<tensor<256xf32>>
  %214 = util.global.address @"__iree_flow___sm_node898__m.layer-117.moving_mean" : !util.ptr<tensor<256xf32>>
  %215 = util.global.address @"__iree_flow___sm_node899__m.layer-117.moving_variance" : !util.ptr<tensor<256xf32>>
  %216 = util.global.address @"__iree_flow___sm_node908__m.layer-119.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %217 = util.global.address @"__iree_flow___sm_node909__m.layer-119.bias" : !util.ptr<tensor<1024xf32>>
  %218 = util.global.address @"__iree_flow___sm_node915__m.layer-120.gamma" : !util.ptr<tensor<1024xf32>>
  %219 = util.global.address @"__iree_flow___sm_node916__m.layer-120.beta" : !util.ptr<tensor<1024xf32>>
  %220 = util.global.address @"__iree_flow___sm_node917__m.layer-120.moving_mean" : !util.ptr<tensor<1024xf32>>
  %221 = util.global.address @"__iree_flow___sm_node918__m.layer-120.moving_variance" : !util.ptr<tensor<1024xf32>>
  %222 = util.global.address @"__iree_flow___sm_node931__m.layer-123.kernel" : !util.ptr<tensor<1x1x1024x256xf32>>
  %223 = util.global.address @"__iree_flow___sm_node932__m.layer-123.bias" : !util.ptr<tensor<256xf32>>
  %224 = util.global.address @"__iree_flow___sm_node938__m.layer-124.gamma" : !util.ptr<tensor<256xf32>>
  %225 = util.global.address @"__iree_flow___sm_node939__m.layer-124.beta" : !util.ptr<tensor<256xf32>>
  %226 = util.global.address @"__iree_flow___sm_node940__m.layer-124.moving_mean" : !util.ptr<tensor<256xf32>>
  %227 = util.global.address @"__iree_flow___sm_node941__m.layer-124.moving_variance" : !util.ptr<tensor<256xf32>>
  %228 = util.global.address @"__iree_flow___sm_node950__m.layer-126.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %229 = util.global.address @"__iree_flow___sm_node951__m.layer-126.bias" : !util.ptr<tensor<256xf32>>
  %230 = util.global.address @"__iree_flow___sm_node957__m.layer-127.gamma" : !util.ptr<tensor<256xf32>>
  %231 = util.global.address @"__iree_flow___sm_node958__m.layer-127.beta" : !util.ptr<tensor<256xf32>>
  %232 = util.global.address @"__iree_flow___sm_node959__m.layer-127.moving_mean" : !util.ptr<tensor<256xf32>>
  %233 = util.global.address @"__iree_flow___sm_node960__m.layer-127.moving_variance" : !util.ptr<tensor<256xf32>>
  %234 = util.global.address @"__iree_flow___sm_node969__m.layer-129.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %235 = util.global.address @"__iree_flow___sm_node970__m.layer-129.bias" : !util.ptr<tensor<1024xf32>>
  %236 = util.global.address @"__iree_flow___sm_node976__m.layer-130.gamma" : !util.ptr<tensor<1024xf32>>
  %237 = util.global.address @"__iree_flow___sm_node977__m.layer-130.beta" : !util.ptr<tensor<1024xf32>>
  %238 = util.global.address @"__iree_flow___sm_node978__m.layer-130.moving_mean" : !util.ptr<tensor<1024xf32>>
  %239 = util.global.address @"__iree_flow___sm_node979__m.layer-130.moving_variance" : !util.ptr<tensor<1024xf32>>
  %240 = util.global.address @"__iree_flow___sm_node992__m.layer-133.kernel" : !util.ptr<tensor<1x1x1024x256xf32>>
  %241 = util.global.address @"__iree_flow___sm_node993__m.layer-133.bias" : !util.ptr<tensor<256xf32>>
  %242 = util.global.address @"__iree_flow___sm_node999__m.layer-134.gamma" : !util.ptr<tensor<256xf32>>
  %243 = util.global.address @"__iree_flow___sm_node1000__m.layer-134.beta" : !util.ptr<tensor<256xf32>>
  %244 = util.global.address @"__iree_flow___sm_node1001__m.layer-134.moving_mean" : !util.ptr<tensor<256xf32>>
  %245 = util.global.address @"__iree_flow___sm_node1002__m.layer-134.moving_variance" : !util.ptr<tensor<256xf32>>
  %246 = util.global.address @"__iree_flow___sm_node1011__m.layer-136.kernel" : !util.ptr<tensor<3x3x256x256xf32>>
  %247 = util.global.address @"__iree_flow___sm_node1012__m.layer-136.bias" : !util.ptr<tensor<256xf32>>
  %248 = util.global.address @"__iree_flow___sm_node1018__m.layer-137.gamma" : !util.ptr<tensor<256xf32>>
  %249 = util.global.address @"__iree_flow___sm_node1019__m.layer-137.beta" : !util.ptr<tensor<256xf32>>
  %250 = util.global.address @"__iree_flow___sm_node1020__m.layer-137.moving_mean" : !util.ptr<tensor<256xf32>>
  %251 = util.global.address @"__iree_flow___sm_node1021__m.layer-137.moving_variance" : !util.ptr<tensor<256xf32>>
  %252 = util.global.address @"__iree_flow___sm_node1030__m.layer-139.kernel" : !util.ptr<tensor<1x1x256x1024xf32>>
  %253 = util.global.address @"__iree_flow___sm_node1031__m.layer-139.bias" : !util.ptr<tensor<1024xf32>>
  %254 = util.global.address @"__iree_flow___sm_node1037__m.layer-140.gamma" : !util.ptr<tensor<1024xf32>>
  %255 = util.global.address @"__iree_flow___sm_node1038__m.layer-140.beta" : !util.ptr<tensor<1024xf32>>
  %256 = util.global.address @"__iree_flow___sm_node1039__m.layer-140.moving_mean" : !util.ptr<tensor<1024xf32>>
  %257 = util.global.address @"__iree_flow___sm_node1040__m.layer-140.moving_variance" : !util.ptr<tensor<1024xf32>>
  %258 = util.global.address @"__iree_flow___sm_node1053__m.layer-143.kernel" : !util.ptr<tensor<1x1x1024x512xf32>>
  %259 = util.global.address @"__iree_flow___sm_node1054__m.layer-143.bias" : !util.ptr<tensor<512xf32>>
  %260 = util.global.address @"__iree_flow___sm_node1060__m.layer-144.gamma" : !util.ptr<tensor<512xf32>>
  %261 = util.global.address @"__iree_flow___sm_node1061__m.layer-144.beta" : !util.ptr<tensor<512xf32>>
  %262 = util.global.address @"__iree_flow___sm_node1062__m.layer-144.moving_mean" : !util.ptr<tensor<512xf32>>
  %263 = util.global.address @"__iree_flow___sm_node1063__m.layer-144.moving_variance" : !util.ptr<tensor<512xf32>>
  %264 = util.global.address @"__iree_flow___sm_node1072__m.layer-146.kernel" : !util.ptr<tensor<3x3x512x512xf32>>
  %265 = util.global.address @"__iree_flow___sm_node1073__m.layer-146.bias" : !util.ptr<tensor<512xf32>>
  %266 = util.global.address @"__iree_flow___sm_node1079__m.layer-147.gamma" : !util.ptr<tensor<512xf32>>
  %267 = util.global.address @"__iree_flow___sm_node1080__m.layer-147.beta" : !util.ptr<tensor<512xf32>>
  %268 = util.global.address @"__iree_flow___sm_node1081__m.layer-147.moving_mean" : !util.ptr<tensor<512xf32>>
  %269 = util.global.address @"__iree_flow___sm_node1082__m.layer-147.moving_variance" : !util.ptr<tensor<512xf32>>
  %270 = util.global.address @"__iree_flow___sm_node1097__m.layer-150.kernel" : !util.ptr<tensor<1x1x512x2048xf32>>
  %271 = util.global.address @"__iree_flow___sm_node1098__m.layer-150.bias" : !util.ptr<tensor<2048xf32>>
  %272 = util.global.address @"__iree_flow___sm_node1091__m.layer-149.kernel" : !util.ptr<tensor<1x1x1024x2048xf32>>
  %273 = util.global.address @"__iree_flow___sm_node1092__m.layer-149.bias" : !util.ptr<tensor<2048xf32>>
  %274 = util.global.address @"__iree_flow___sm_node1104__m.layer-151.gamma" : !util.ptr<tensor<2048xf32>>
  %275 = util.global.address @"__iree_flow___sm_node1105__m.layer-151.beta" : !util.ptr<tensor<2048xf32>>
  %276 = util.global.address @"__iree_flow___sm_node1106__m.layer-151.moving_mean" : !util.ptr<tensor<2048xf32>>
  %277 = util.global.address @"__iree_flow___sm_node1107__m.layer-151.moving_variance" : !util.ptr<tensor<2048xf32>>
  %278 = util.global.address @"__iree_flow___sm_node1113__m.layer-152.gamma" : !util.ptr<tensor<2048xf32>>
  %279 = util.global.address @"__iree_flow___sm_node1114__m.layer-152.beta" : !util.ptr<tensor<2048xf32>>
  %280 = util.global.address @"__iree_flow___sm_node1115__m.layer-152.moving_mean" : !util.ptr<tensor<2048xf32>>
  %281 = util.global.address @"__iree_flow___sm_node1116__m.layer-152.moving_variance" : !util.ptr<tensor<2048xf32>>
  %282 = util.global.address @"__iree_flow___sm_node1129__m.layer-155.kernel" : !util.ptr<tensor<1x1x2048x512xf32>>
  %283 = util.global.address @"__iree_flow___sm_node1130__m.layer-155.bias" : !util.ptr<tensor<512xf32>>
  %284 = util.global.address @"__iree_flow___sm_node1136__m.layer-156.gamma" : !util.ptr<tensor<512xf32>>
  %285 = util.global.address @"__iree_flow___sm_node1137__m.layer-156.beta" : !util.ptr<tensor<512xf32>>
  %286 = util.global.address @"__iree_flow___sm_node1138__m.layer-156.moving_mean" : !util.ptr<tensor<512xf32>>
  %287 = util.global.address @"__iree_flow___sm_node1139__m.layer-156.moving_variance" : !util.ptr<tensor<512xf32>>
  %288 = util.global.address @"__iree_flow___sm_node1148__m.layer-158.kernel" : !util.ptr<tensor<3x3x512x512xf32>>
  %289 = util.global.address @"__iree_flow___sm_node1149__m.layer-158.bias" : !util.ptr<tensor<512xf32>>
  %290 = util.global.address @"__iree_flow___sm_node1155__m.layer-159.gamma" : !util.ptr<tensor<512xf32>>
  %291 = util.global.address @"__iree_flow___sm_node1156__m.layer-159.beta" : !util.ptr<tensor<512xf32>>
  %292 = util.global.address @"__iree_flow___sm_node1157__m.layer-159.moving_mean" : !util.ptr<tensor<512xf32>>
  %293 = util.global.address @"__iree_flow___sm_node1158__m.layer-159.moving_variance" : !util.ptr<tensor<512xf32>>
  %294 = util.global.address @"__iree_flow___sm_node1167__m.layer-161.kernel" : !util.ptr<tensor<1x1x512x2048xf32>>
  %295 = util.global.address @"__iree_flow___sm_node1168__m.layer-161.bias" : !util.ptr<tensor<2048xf32>>
  %296 = util.global.address @"__iree_flow___sm_node1174__m.layer-162.gamma" : !util.ptr<tensor<2048xf32>>
  %297 = util.global.address @"__iree_flow___sm_node1175__m.layer-162.beta" : !util.ptr<tensor<2048xf32>>
  %298 = util.global.address @"__iree_flow___sm_node1176__m.layer-162.moving_mean" : !util.ptr<tensor<2048xf32>>
  %299 = util.global.address @"__iree_flow___sm_node1177__m.layer-162.moving_variance" : !util.ptr<tensor<2048xf32>>
  %300 = util.global.address @"__iree_flow___sm_node1190__m.layer-165.kernel" : !util.ptr<tensor<1x1x2048x512xf32>>
  %301 = util.global.address @"__iree_flow___sm_node1191__m.layer-165.bias" : !util.ptr<tensor<512xf32>>
  %302 = util.global.address @"__iree_flow___sm_node1197__m.layer-166.gamma" : !util.ptr<tensor<512xf32>>
  %303 = util.global.address @"__iree_flow___sm_node1198__m.layer-166.beta" : !util.ptr<tensor<512xf32>>
  %304 = util.global.address @"__iree_flow___sm_node1199__m.layer-166.moving_mean" : !util.ptr<tensor<512xf32>>
  %305 = util.global.address @"__iree_flow___sm_node1200__m.layer-166.moving_variance" : !util.ptr<tensor<512xf32>>
  %306 = util.global.address @"__iree_flow___sm_node1209__m.layer-168.kernel" : !util.ptr<tensor<3x3x512x512xf32>>
  %307 = util.global.address @"__iree_flow___sm_node1210__m.layer-168.bias" : !util.ptr<tensor<512xf32>>
  %308 = util.global.address @"__iree_flow___sm_node1216__m.layer-169.gamma" : !util.ptr<tensor<512xf32>>
  %309 = util.global.address @"__iree_flow___sm_node1217__m.layer-169.beta" : !util.ptr<tensor<512xf32>>
  %310 = util.global.address @"__iree_flow___sm_node1218__m.layer-169.moving_mean" : !util.ptr<tensor<512xf32>>
  %311 = util.global.address @"__iree_flow___sm_node1219__m.layer-169.moving_variance" : !util.ptr<tensor<512xf32>>
  %312 = util.global.address @"__iree_flow___sm_node1228__m.layer-171.kernel" : !util.ptr<tensor<1x1x512x2048xf32>>
  %313 = util.global.address @"__iree_flow___sm_node1229__m.layer-171.bias" : !util.ptr<tensor<2048xf32>>
  %314 = util.global.address @"__iree_flow___sm_node1235__m.layer-172.gamma" : !util.ptr<tensor<2048xf32>>
  %315 = util.global.address @"__iree_flow___sm_node1236__m.layer-172.beta" : !util.ptr<tensor<2048xf32>>
  %316 = util.global.address @"__iree_flow___sm_node1237__m.layer-172.moving_mean" : !util.ptr<tensor<2048xf32>>
  %317 = util.global.address @"__iree_flow___sm_node1238__m.layer-172.moving_variance" : !util.ptr<tensor<2048xf32>>
  %318 = util.global.address @"__iree_flow___sm_node1255__m.layer-176.kernel" : !util.ptr<tensor<2048x1000xf32>>
  %319 = util.global.address @"__iree_flow___sm_node1256__m.layer-176.bias" : !util.ptr<tensor<1000xf32>>
  %320 = mhlo.constant dense<0.000000e+00> : tensor<1x112x112x64xf32>
  %321 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %322 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x256xf32>
  %323 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x128xf32>
  %324 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x512xf32>
  %325 = mhlo.constant dense<0.000000e+00> : tensor<1x14x14x256xf32>
  %326 = mhlo.constant dense<0.000000e+00> : tensor<1x14x14x1024xf32>
  %327 = mhlo.constant dense<0.000000e+00> : tensor<1x7x7x512xf32>
  %328 = mhlo.constant dense<0.000000e+00> : tensor<1x7x7x2048xf32>
  %329 = mhlo.constant dense<4.900000e+01> : tensor<1x2048xf32>
  %330 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %331 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %332 = util.global.load.indirect %5 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %333 = util.global.load.indirect %4 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %334 = util.global.load.indirect %3 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %335 = util.global.load.indirect %2 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %336 = util.global.load.indirect %1 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %337 = util.global.load.indirect %0 : !util.ptr<tensor<7x7x3x64xf32>> -> tensor<7x7x3x64xf32>
  %338 = util.global.load.indirect %25 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %339 = util.global.load.indirect %24 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %340 = util.global.load.indirect %23 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %341 = util.global.load.indirect %22 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %342 = util.global.load.indirect %21 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %343 = util.global.load.indirect %20 : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %344 = util.global.load.indirect %11 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %345 = util.global.load.indirect %10 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %346 = util.global.load.indirect %9 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %347 = util.global.load.indirect %8 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %348 = util.global.load.indirect %7 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %349 = util.global.load.indirect %6 : !util.ptr<tensor<1x1x64x64xf32>> -> tensor<1x1x64x64xf32>
  %350 = util.global.load.indirect %17 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %351 = util.global.load.indirect %16 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %352 = util.global.load.indirect %15 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %353 = util.global.load.indirect %14 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %354 = util.global.load.indirect %13 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %355 = util.global.load.indirect %12 : !util.ptr<tensor<3x3x64x64xf32>> -> tensor<3x3x64x64xf32>
  %356 = util.global.load.indirect %29 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %357 = util.global.load.indirect %28 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %358 = util.global.load.indirect %27 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %359 = util.global.load.indirect %26 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %360 = util.global.load.indirect %19 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %361 = util.global.load.indirect %18 : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %362 = util.global.load.indirect %35 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %363 = util.global.load.indirect %34 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %364 = util.global.load.indirect %33 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %365 = util.global.load.indirect %32 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %366 = util.global.load.indirect %31 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %367 = util.global.load.indirect %30 : !util.ptr<tensor<1x1x256x64xf32>> -> tensor<1x1x256x64xf32>
  %368 = util.global.load.indirect %41 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %369 = util.global.load.indirect %40 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %370 = util.global.load.indirect %39 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %371 = util.global.load.indirect %38 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %372 = util.global.load.indirect %37 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %373 = util.global.load.indirect %36 : !util.ptr<tensor<3x3x64x64xf32>> -> tensor<3x3x64x64xf32>
  %374 = util.global.load.indirect %47 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %375 = util.global.load.indirect %46 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %376 = util.global.load.indirect %45 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %377 = util.global.load.indirect %44 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %378 = util.global.load.indirect %43 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %379 = util.global.load.indirect %42 : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %380 = util.global.load.indirect %53 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %381 = util.global.load.indirect %52 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %382 = util.global.load.indirect %51 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %383 = util.global.load.indirect %50 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %384 = util.global.load.indirect %49 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %385 = util.global.load.indirect %48 : !util.ptr<tensor<1x1x256x64xf32>> -> tensor<1x1x256x64xf32>
  %386 = util.global.load.indirect %59 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %387 = util.global.load.indirect %58 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %388 = util.global.load.indirect %57 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %389 = util.global.load.indirect %56 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %390 = util.global.load.indirect %55 : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %391 = util.global.load.indirect %54 : !util.ptr<tensor<3x3x64x64xf32>> -> tensor<3x3x64x64xf32>
  %392 = util.global.load.indirect %65 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %393 = util.global.load.indirect %64 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %394 = util.global.load.indirect %63 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %395 = util.global.load.indirect %62 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %396 = util.global.load.indirect %61 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %397 = util.global.load.indirect %60 : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %398 = util.global.load.indirect %85 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %399 = util.global.load.indirect %84 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %400 = util.global.load.indirect %83 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %401 = util.global.load.indirect %82 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %402 = util.global.load.indirect %81 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %403 = util.global.load.indirect %80 : !util.ptr<tensor<1x1x256x512xf32>> -> tensor<1x1x256x512xf32>
  %404 = util.global.load.indirect %71 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %405 = util.global.load.indirect %70 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %406 = util.global.load.indirect %69 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %407 = util.global.load.indirect %68 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %408 = util.global.load.indirect %67 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %409 = util.global.load.indirect %66 : !util.ptr<tensor<1x1x256x128xf32>> -> tensor<1x1x256x128xf32>
  %410 = util.global.load.indirect %77 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %411 = util.global.load.indirect %76 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %412 = util.global.load.indirect %75 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %413 = util.global.load.indirect %74 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %414 = util.global.load.indirect %73 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %415 = util.global.load.indirect %72 : !util.ptr<tensor<3x3x128x128xf32>> -> tensor<3x3x128x128xf32>
  %416 = util.global.load.indirect %89 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %417 = util.global.load.indirect %88 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %418 = util.global.load.indirect %87 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %419 = util.global.load.indirect %86 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %420 = util.global.load.indirect %79 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %421 = util.global.load.indirect %78 : !util.ptr<tensor<1x1x128x512xf32>> -> tensor<1x1x128x512xf32>
  %422 = util.global.load.indirect %95 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %423 = util.global.load.indirect %94 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %424 = util.global.load.indirect %93 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %425 = util.global.load.indirect %92 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %426 = util.global.load.indirect %91 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %427 = util.global.load.indirect %90 : !util.ptr<tensor<1x1x512x128xf32>> -> tensor<1x1x512x128xf32>
  %428 = util.global.load.indirect %101 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %429 = util.global.load.indirect %100 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %430 = util.global.load.indirect %99 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %431 = util.global.load.indirect %98 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %432 = util.global.load.indirect %97 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %433 = util.global.load.indirect %96 : !util.ptr<tensor<3x3x128x128xf32>> -> tensor<3x3x128x128xf32>
  %434 = util.global.load.indirect %107 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %435 = util.global.load.indirect %106 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %436 = util.global.load.indirect %105 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %437 = util.global.load.indirect %104 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %438 = util.global.load.indirect %103 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %439 = util.global.load.indirect %102 : !util.ptr<tensor<1x1x128x512xf32>> -> tensor<1x1x128x512xf32>
  %440 = util.global.load.indirect %113 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %441 = util.global.load.indirect %112 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %442 = util.global.load.indirect %111 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %443 = util.global.load.indirect %110 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %444 = util.global.load.indirect %109 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %445 = util.global.load.indirect %108 : !util.ptr<tensor<1x1x512x128xf32>> -> tensor<1x1x512x128xf32>
  %446 = util.global.load.indirect %119 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %447 = util.global.load.indirect %118 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %448 = util.global.load.indirect %117 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %449 = util.global.load.indirect %116 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %450 = util.global.load.indirect %115 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %451 = util.global.load.indirect %114 : !util.ptr<tensor<3x3x128x128xf32>> -> tensor<3x3x128x128xf32>
  %452 = util.global.load.indirect %125 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %453 = util.global.load.indirect %124 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %454 = util.global.load.indirect %123 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %455 = util.global.load.indirect %122 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %456 = util.global.load.indirect %121 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %457 = util.global.load.indirect %120 : !util.ptr<tensor<1x1x128x512xf32>> -> tensor<1x1x128x512xf32>
  %458 = util.global.load.indirect %131 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %459 = util.global.load.indirect %130 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %460 = util.global.load.indirect %129 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %461 = util.global.load.indirect %128 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %462 = util.global.load.indirect %127 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %463 = util.global.load.indirect %126 : !util.ptr<tensor<1x1x512x128xf32>> -> tensor<1x1x512x128xf32>
  %464 = util.global.load.indirect %137 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %465 = util.global.load.indirect %136 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %466 = util.global.load.indirect %135 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %467 = util.global.load.indirect %134 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %468 = util.global.load.indirect %133 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  %469 = util.global.load.indirect %132 : !util.ptr<tensor<3x3x128x128xf32>> -> tensor<3x3x128x128xf32>
  %470 = util.global.load.indirect %143 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %471 = util.global.load.indirect %142 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %472 = util.global.load.indirect %141 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %473 = util.global.load.indirect %140 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %474 = util.global.load.indirect %139 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %475 = util.global.load.indirect %138 : !util.ptr<tensor<1x1x128x512xf32>> -> tensor<1x1x128x512xf32>
  %476 = util.global.load.indirect %163 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %477 = util.global.load.indirect %162 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %478 = util.global.load.indirect %161 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %479 = util.global.load.indirect %160 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %480 = util.global.load.indirect %159 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %481 = util.global.load.indirect %158 : !util.ptr<tensor<1x1x512x1024xf32>> -> tensor<1x1x512x1024xf32>
  %482 = util.global.load.indirect %149 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %483 = util.global.load.indirect %148 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %484 = util.global.load.indirect %147 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %485 = util.global.load.indirect %146 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %486 = util.global.load.indirect %145 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %487 = util.global.load.indirect %144 : !util.ptr<tensor<1x1x512x256xf32>> -> tensor<1x1x512x256xf32>
  %488 = util.global.load.indirect %155 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %489 = util.global.load.indirect %154 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %490 = util.global.load.indirect %153 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %491 = util.global.load.indirect %152 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %492 = util.global.load.indirect %151 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %493 = util.global.load.indirect %150 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %494 = util.global.load.indirect %167 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %495 = util.global.load.indirect %166 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %496 = util.global.load.indirect %165 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %497 = util.global.load.indirect %164 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %498 = util.global.load.indirect %157 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %499 = util.global.load.indirect %156 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %500 = util.global.load.indirect %173 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %501 = util.global.load.indirect %172 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %502 = util.global.load.indirect %171 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %503 = util.global.load.indirect %170 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %504 = util.global.load.indirect %169 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %505 = util.global.load.indirect %168 : !util.ptr<tensor<1x1x1024x256xf32>> -> tensor<1x1x1024x256xf32>
  %506 = util.global.load.indirect %179 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %507 = util.global.load.indirect %178 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %508 = util.global.load.indirect %177 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %509 = util.global.load.indirect %176 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %510 = util.global.load.indirect %175 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %511 = util.global.load.indirect %174 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %512 = util.global.load.indirect %185 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %513 = util.global.load.indirect %184 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %514 = util.global.load.indirect %183 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %515 = util.global.load.indirect %182 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %516 = util.global.load.indirect %181 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %517 = util.global.load.indirect %180 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %518 = util.global.load.indirect %191 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %519 = util.global.load.indirect %190 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %520 = util.global.load.indirect %189 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %521 = util.global.load.indirect %188 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %522 = util.global.load.indirect %187 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %523 = util.global.load.indirect %186 : !util.ptr<tensor<1x1x1024x256xf32>> -> tensor<1x1x1024x256xf32>
  %524 = util.global.load.indirect %197 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %525 = util.global.load.indirect %196 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %526 = util.global.load.indirect %195 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %527 = util.global.load.indirect %194 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %528 = util.global.load.indirect %193 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %529 = util.global.load.indirect %192 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %530 = util.global.load.indirect %203 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %531 = util.global.load.indirect %202 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %532 = util.global.load.indirect %201 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %533 = util.global.load.indirect %200 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %534 = util.global.load.indirect %199 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %535 = util.global.load.indirect %198 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %536 = util.global.load.indirect %209 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %537 = util.global.load.indirect %208 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %538 = util.global.load.indirect %207 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %539 = util.global.load.indirect %206 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %540 = util.global.load.indirect %205 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %541 = util.global.load.indirect %204 : !util.ptr<tensor<1x1x1024x256xf32>> -> tensor<1x1x1024x256xf32>
  %542 = util.global.load.indirect %215 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %543 = util.global.load.indirect %214 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %544 = util.global.load.indirect %213 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %545 = util.global.load.indirect %212 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %546 = util.global.load.indirect %211 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %547 = util.global.load.indirect %210 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %548 = util.global.load.indirect %221 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %549 = util.global.load.indirect %220 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %550 = util.global.load.indirect %219 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %551 = util.global.load.indirect %218 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %552 = util.global.load.indirect %217 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %553 = util.global.load.indirect %216 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %554 = util.global.load.indirect %227 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %555 = util.global.load.indirect %226 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %556 = util.global.load.indirect %225 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %557 = util.global.load.indirect %224 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %558 = util.global.load.indirect %223 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %559 = util.global.load.indirect %222 : !util.ptr<tensor<1x1x1024x256xf32>> -> tensor<1x1x1024x256xf32>
  %560 = util.global.load.indirect %233 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %561 = util.global.load.indirect %232 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %562 = util.global.load.indirect %231 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %563 = util.global.load.indirect %230 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %564 = util.global.load.indirect %229 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %565 = util.global.load.indirect %228 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %566 = util.global.load.indirect %239 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %567 = util.global.load.indirect %238 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %568 = util.global.load.indirect %237 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %569 = util.global.load.indirect %236 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %570 = util.global.load.indirect %235 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %571 = util.global.load.indirect %234 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %572 = util.global.load.indirect %245 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %573 = util.global.load.indirect %244 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %574 = util.global.load.indirect %243 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %575 = util.global.load.indirect %242 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %576 = util.global.load.indirect %241 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %577 = util.global.load.indirect %240 : !util.ptr<tensor<1x1x1024x256xf32>> -> tensor<1x1x1024x256xf32>
  %578 = util.global.load.indirect %251 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %579 = util.global.load.indirect %250 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %580 = util.global.load.indirect %249 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %581 = util.global.load.indirect %248 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %582 = util.global.load.indirect %247 : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %583 = util.global.load.indirect %246 : !util.ptr<tensor<3x3x256x256xf32>> -> tensor<3x3x256x256xf32>
  %584 = util.global.load.indirect %257 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %585 = util.global.load.indirect %256 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %586 = util.global.load.indirect %255 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %587 = util.global.load.indirect %254 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %588 = util.global.load.indirect %253 : !util.ptr<tensor<1024xf32>> -> tensor<1024xf32>
  %589 = util.global.load.indirect %252 : !util.ptr<tensor<1x1x256x1024xf32>> -> tensor<1x1x256x1024xf32>
  %590 = util.global.load.indirect %277 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %591 = util.global.load.indirect %276 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %592 = util.global.load.indirect %275 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %593 = util.global.load.indirect %274 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %594 = util.global.load.indirect %273 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %595 = util.global.load.indirect %272 : !util.ptr<tensor<1x1x1024x2048xf32>> -> tensor<1x1x1024x2048xf32>
  %596 = util.global.load.indirect %263 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %597 = util.global.load.indirect %262 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %598 = util.global.load.indirect %261 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %599 = util.global.load.indirect %260 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %600 = util.global.load.indirect %259 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %601 = util.global.load.indirect %258 : !util.ptr<tensor<1x1x1024x512xf32>> -> tensor<1x1x1024x512xf32>
  %602 = util.global.load.indirect %269 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %603 = util.global.load.indirect %268 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %604 = util.global.load.indirect %267 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %605 = util.global.load.indirect %266 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %606 = util.global.load.indirect %265 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %607 = util.global.load.indirect %264 : !util.ptr<tensor<3x3x512x512xf32>> -> tensor<3x3x512x512xf32>
  %608 = util.global.load.indirect %281 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %609 = util.global.load.indirect %280 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %610 = util.global.load.indirect %279 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %611 = util.global.load.indirect %278 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %612 = util.global.load.indirect %271 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %613 = util.global.load.indirect %270 : !util.ptr<tensor<1x1x512x2048xf32>> -> tensor<1x1x512x2048xf32>
  %614 = util.global.load.indirect %287 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %615 = util.global.load.indirect %286 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %616 = util.global.load.indirect %285 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %617 = util.global.load.indirect %284 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %618 = util.global.load.indirect %283 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %619 = util.global.load.indirect %282 : !util.ptr<tensor<1x1x2048x512xf32>> -> tensor<1x1x2048x512xf32>
  %620 = util.global.load.indirect %293 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %621 = util.global.load.indirect %292 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %622 = util.global.load.indirect %291 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %623 = util.global.load.indirect %290 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %624 = util.global.load.indirect %289 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %625 = util.global.load.indirect %288 : !util.ptr<tensor<3x3x512x512xf32>> -> tensor<3x3x512x512xf32>
  %626 = util.global.load.indirect %299 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %627 = util.global.load.indirect %298 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %628 = util.global.load.indirect %297 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %629 = util.global.load.indirect %296 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %630 = util.global.load.indirect %295 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %631 = util.global.load.indirect %294 : !util.ptr<tensor<1x1x512x2048xf32>> -> tensor<1x1x512x2048xf32>
  %632 = util.global.load.indirect %305 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %633 = util.global.load.indirect %304 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %634 = util.global.load.indirect %303 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %635 = util.global.load.indirect %302 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %636 = util.global.load.indirect %301 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %637 = util.global.load.indirect %300 : !util.ptr<tensor<1x1x2048x512xf32>> -> tensor<1x1x2048x512xf32>
  %638 = util.global.load.indirect %311 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %639 = util.global.load.indirect %310 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %640 = util.global.load.indirect %309 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %641 = util.global.load.indirect %308 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %642 = util.global.load.indirect %307 : !util.ptr<tensor<512xf32>> -> tensor<512xf32>
  %643 = util.global.load.indirect %306 : !util.ptr<tensor<3x3x512x512xf32>> -> tensor<3x3x512x512xf32>
  %644 = util.global.load.indirect %317 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %645 = util.global.load.indirect %316 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %646 = util.global.load.indirect %315 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %647 = util.global.load.indirect %314 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %648 = util.global.load.indirect %313 : !util.ptr<tensor<2048xf32>> -> tensor<2048xf32>
  %649 = util.global.load.indirect %312 : !util.ptr<tensor<1x1x512x2048xf32>> -> tensor<1x1x512x2048xf32>
  %650 = util.global.load.indirect %319 : !util.ptr<tensor<1000xf32>> -> tensor<1000xf32>
  %651 = util.global.load.indirect %318 : !util.ptr<tensor<2048x1000xf32>> -> tensor<2048x1000xf32>
  %652 = "mhlo.pad"(%arg0, %331) {edge_padding_high = dense<[0, 3, 3, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 3, 3, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x224x224x3xf32>, tensor<f32>) -> tensor<1x230x230x3xf32>
  %653 = "mhlo.convolution"(%652, %337) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
  %654 = "mhlo.broadcast_in_dim"(%336) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %655 = mhlo.add %653, %654 : tensor<1x112x112x64xf32>
  %656 = "mhlo.batch_norm_inference"(%655, %335, %334, %333, %332) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x112x112x64xf32>
  %657 = mhlo.maximum %656, %320 : tensor<1x112x112x64xf32>
  %658 = "mhlo.pad"(%657, %331) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x112x112x64xf32>, tensor<f32>) -> tensor<1x114x114x64xf32>
  %659 = "mhlo.reduce_window"(%658, %330) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %944 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%944) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x114x114x64xf32>, tensor<f32>) -> tensor<1x56x56x64xf32>
  %660 = "mhlo.convolution"(%659, %343) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  %661 = "mhlo.broadcast_in_dim"(%342) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %662 = mhlo.add %660, %661 : tensor<1x56x56x256xf32>
  %663 = "mhlo.batch_norm_inference"(%662, %341, %340, %339, %338) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %664 = "mhlo.convolution"(%659, %349) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32>
  %665 = "mhlo.broadcast_in_dim"(%348) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %666 = mhlo.add %664, %665 : tensor<1x56x56x64xf32>
  %667 = "mhlo.batch_norm_inference"(%666, %347, %346, %345, %344) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %668 = mhlo.maximum %667, %321 : tensor<1x56x56x64xf32>
  %669 = "mhlo.convolution"(%668, %355) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
  %670 = "mhlo.broadcast_in_dim"(%354) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %671 = mhlo.add %669, %670 : tensor<1x56x56x64xf32>
  %672 = "mhlo.batch_norm_inference"(%671, %353, %352, %351, %350) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %673 = mhlo.maximum %672, %321 : tensor<1x56x56x64xf32>
  %674 = "mhlo.convolution"(%673, %361) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  %675 = "mhlo.broadcast_in_dim"(%360) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %676 = mhlo.add %674, %675 : tensor<1x56x56x256xf32>
  %677 = "mhlo.batch_norm_inference"(%676, %359, %358, %357, %356) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %678 = mhlo.add %663, %677 : tensor<1x56x56x256xf32>
  %679 = mhlo.maximum %678, %322 : tensor<1x56x56x256xf32>
  %680 = "mhlo.convolution"(%679, %367) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
  %681 = "mhlo.broadcast_in_dim"(%366) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %682 = mhlo.add %680, %681 : tensor<1x56x56x64xf32>
  %683 = "mhlo.batch_norm_inference"(%682, %365, %364, %363, %362) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %684 = mhlo.maximum %683, %321 : tensor<1x56x56x64xf32>
  %685 = "mhlo.convolution"(%684, %373) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
  %686 = "mhlo.broadcast_in_dim"(%372) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %687 = mhlo.add %685, %686 : tensor<1x56x56x64xf32>
  %688 = "mhlo.batch_norm_inference"(%687, %371, %370, %369, %368) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %689 = mhlo.maximum %688, %321 : tensor<1x56x56x64xf32>
  %690 = "mhlo.convolution"(%689, %379) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  %691 = "mhlo.broadcast_in_dim"(%378) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %692 = mhlo.add %690, %691 : tensor<1x56x56x256xf32>
  %693 = "mhlo.batch_norm_inference"(%692, %377, %376, %375, %374) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %694 = mhlo.add %679, %693 : tensor<1x56x56x256xf32>
  %695 = mhlo.maximum %694, %322 : tensor<1x56x56x256xf32>
  %696 = "mhlo.convolution"(%695, %385) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x256xf32>, tensor<1x1x256x64xf32>) -> tensor<1x56x56x64xf32>
  %697 = "mhlo.broadcast_in_dim"(%384) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %698 = mhlo.add %696, %697 : tensor<1x56x56x64xf32>
  %699 = "mhlo.batch_norm_inference"(%698, %383, %382, %381, %380) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %700 = mhlo.maximum %699, %321 : tensor<1x56x56x64xf32>
  %701 = "mhlo.convolution"(%700, %391) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
  %702 = "mhlo.broadcast_in_dim"(%390) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %703 = mhlo.add %701, %702 : tensor<1x56x56x64xf32>
  %704 = "mhlo.batch_norm_inference"(%703, %389, %388, %387, %386) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %705 = mhlo.maximum %704, %321 : tensor<1x56x56x64xf32>
  %706 = "mhlo.convolution"(%705, %397) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  %707 = "mhlo.broadcast_in_dim"(%396) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %708 = mhlo.add %706, %707 : tensor<1x56x56x256xf32>
  %709 = "mhlo.batch_norm_inference"(%708, %395, %394, %393, %392) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %710 = mhlo.add %695, %709 : tensor<1x56x56x256xf32>
  %711 = mhlo.maximum %710, %322 : tensor<1x56x56x256xf32>
  %712 = "mhlo.convolution"(%711, %403) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x56x56x256xf32>, tensor<1x1x256x512xf32>) -> tensor<1x28x28x512xf32>
  %713 = "mhlo.broadcast_in_dim"(%402) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %714 = mhlo.add %712, %713 : tensor<1x28x28x512xf32>
  %715 = "mhlo.batch_norm_inference"(%714, %401, %400, %399, %398) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %716 = "mhlo.convolution"(%711, %409) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x56x56x256xf32>, tensor<1x1x256x128xf32>) -> tensor<1x28x28x128xf32>
  %717 = "mhlo.broadcast_in_dim"(%408) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %718 = mhlo.add %716, %717 : tensor<1x28x28x128xf32>
  %719 = "mhlo.batch_norm_inference"(%718, %407, %406, %405, %404) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %720 = mhlo.maximum %719, %323 : tensor<1x28x28x128xf32>
  %721 = "mhlo.convolution"(%720, %415) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
  %722 = "mhlo.broadcast_in_dim"(%414) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %723 = mhlo.add %721, %722 : tensor<1x28x28x128xf32>
  %724 = "mhlo.batch_norm_inference"(%723, %413, %412, %411, %410) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %725 = mhlo.maximum %724, %323 : tensor<1x28x28x128xf32>
  %726 = "mhlo.convolution"(%725, %421) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
  %727 = "mhlo.broadcast_in_dim"(%420) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %728 = mhlo.add %726, %727 : tensor<1x28x28x512xf32>
  %729 = "mhlo.batch_norm_inference"(%728, %419, %418, %417, %416) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %730 = mhlo.add %715, %729 : tensor<1x28x28x512xf32>
  %731 = mhlo.maximum %730, %324 : tensor<1x28x28x512xf32>
  %732 = "mhlo.convolution"(%731, %427) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
  %733 = "mhlo.broadcast_in_dim"(%426) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %734 = mhlo.add %732, %733 : tensor<1x28x28x128xf32>
  %735 = "mhlo.batch_norm_inference"(%734, %425, %424, %423, %422) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %736 = mhlo.maximum %735, %323 : tensor<1x28x28x128xf32>
  %737 = "mhlo.convolution"(%736, %433) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
  %738 = "mhlo.broadcast_in_dim"(%432) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %739 = mhlo.add %737, %738 : tensor<1x28x28x128xf32>
  %740 = "mhlo.batch_norm_inference"(%739, %431, %430, %429, %428) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %741 = mhlo.maximum %740, %323 : tensor<1x28x28x128xf32>
  %742 = "mhlo.convolution"(%741, %439) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
  %743 = "mhlo.broadcast_in_dim"(%438) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %744 = mhlo.add %742, %743 : tensor<1x28x28x512xf32>
  %745 = "mhlo.batch_norm_inference"(%744, %437, %436, %435, %434) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %746 = mhlo.add %731, %745 : tensor<1x28x28x512xf32>
  %747 = mhlo.maximum %746, %324 : tensor<1x28x28x512xf32>
  %748 = "mhlo.convolution"(%747, %445) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
  %749 = "mhlo.broadcast_in_dim"(%444) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %750 = mhlo.add %748, %749 : tensor<1x28x28x128xf32>
  %751 = "mhlo.batch_norm_inference"(%750, %443, %442, %441, %440) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %752 = mhlo.maximum %751, %323 : tensor<1x28x28x128xf32>
  %753 = "mhlo.convolution"(%752, %451) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
  %754 = "mhlo.broadcast_in_dim"(%450) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %755 = mhlo.add %753, %754 : tensor<1x28x28x128xf32>
  %756 = "mhlo.batch_norm_inference"(%755, %449, %448, %447, %446) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %757 = mhlo.maximum %756, %323 : tensor<1x28x28x128xf32>
  %758 = "mhlo.convolution"(%757, %457) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
  %759 = "mhlo.broadcast_in_dim"(%456) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %760 = mhlo.add %758, %759 : tensor<1x28x28x512xf32>
  %761 = "mhlo.batch_norm_inference"(%760, %455, %454, %453, %452) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %762 = mhlo.add %747, %761 : tensor<1x28x28x512xf32>
  %763 = mhlo.maximum %762, %324 : tensor<1x28x28x512xf32>
  %764 = "mhlo.convolution"(%763, %463) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x512xf32>, tensor<1x1x512x128xf32>) -> tensor<1x28x28x128xf32>
  %765 = "mhlo.broadcast_in_dim"(%462) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %766 = mhlo.add %764, %765 : tensor<1x28x28x128xf32>
  %767 = "mhlo.batch_norm_inference"(%766, %461, %460, %459, %458) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %768 = mhlo.maximum %767, %323 : tensor<1x28x28x128xf32>
  %769 = "mhlo.convolution"(%768, %469) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x28x28x128xf32>
  %770 = "mhlo.broadcast_in_dim"(%468) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %771 = mhlo.add %769, %770 : tensor<1x28x28x128xf32>
  %772 = "mhlo.batch_norm_inference"(%771, %467, %466, %465, %464) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x28x28x128xf32>
  %773 = mhlo.maximum %772, %323 : tensor<1x28x28x128xf32>
  %774 = "mhlo.convolution"(%773, %475) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x128xf32>, tensor<1x1x128x512xf32>) -> tensor<1x28x28x512xf32>
  %775 = "mhlo.broadcast_in_dim"(%474) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %776 = mhlo.add %774, %775 : tensor<1x28x28x512xf32>
  %777 = "mhlo.batch_norm_inference"(%776, %473, %472, %471, %470) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x28x28x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
  %778 = mhlo.add %763, %777 : tensor<1x28x28x512xf32>
  %779 = mhlo.maximum %778, %324 : tensor<1x28x28x512xf32>
  %780 = "mhlo.convolution"(%779, %481) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) -> tensor<1x14x14x1024xf32>
  %781 = "mhlo.broadcast_in_dim"(%480) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %782 = mhlo.add %780, %781 : tensor<1x14x14x1024xf32>
  %783 = "mhlo.batch_norm_inference"(%782, %479, %478, %477, %476) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %784 = "mhlo.convolution"(%779, %487) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x28x28x512xf32>, tensor<1x1x512x256xf32>) -> tensor<1x14x14x256xf32>
  %785 = "mhlo.broadcast_in_dim"(%486) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %786 = mhlo.add %784, %785 : tensor<1x14x14x256xf32>
  %787 = "mhlo.batch_norm_inference"(%786, %485, %484, %483, %482) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %788 = mhlo.maximum %787, %325 : tensor<1x14x14x256xf32>
  %789 = "mhlo.convolution"(%788, %493) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %790 = "mhlo.broadcast_in_dim"(%492) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %791 = mhlo.add %789, %790 : tensor<1x14x14x256xf32>
  %792 = "mhlo.batch_norm_inference"(%791, %491, %490, %489, %488) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %793 = mhlo.maximum %792, %325 : tensor<1x14x14x256xf32>
  %794 = "mhlo.convolution"(%793, %499) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %795 = "mhlo.broadcast_in_dim"(%498) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %796 = mhlo.add %794, %795 : tensor<1x14x14x1024xf32>
  %797 = "mhlo.batch_norm_inference"(%796, %497, %496, %495, %494) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %798 = mhlo.add %783, %797 : tensor<1x14x14x1024xf32>
  %799 = mhlo.maximum %798, %326 : tensor<1x14x14x1024xf32>
  %800 = "mhlo.convolution"(%799, %505) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
  %801 = "mhlo.broadcast_in_dim"(%504) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %802 = mhlo.add %800, %801 : tensor<1x14x14x256xf32>
  %803 = "mhlo.batch_norm_inference"(%802, %503, %502, %501, %500) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %804 = mhlo.maximum %803, %325 : tensor<1x14x14x256xf32>
  %805 = "mhlo.convolution"(%804, %511) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %806 = "mhlo.broadcast_in_dim"(%510) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %807 = mhlo.add %805, %806 : tensor<1x14x14x256xf32>
  %808 = "mhlo.batch_norm_inference"(%807, %509, %508, %507, %506) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %809 = mhlo.maximum %808, %325 : tensor<1x14x14x256xf32>
  %810 = "mhlo.convolution"(%809, %517) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %811 = "mhlo.broadcast_in_dim"(%516) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %812 = mhlo.add %810, %811 : tensor<1x14x14x1024xf32>
  %813 = "mhlo.batch_norm_inference"(%812, %515, %514, %513, %512) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %814 = mhlo.add %799, %813 : tensor<1x14x14x1024xf32>
  %815 = mhlo.maximum %814, %326 : tensor<1x14x14x1024xf32>
  %816 = "mhlo.convolution"(%815, %523) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
  %817 = "mhlo.broadcast_in_dim"(%522) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %818 = mhlo.add %816, %817 : tensor<1x14x14x256xf32>
  %819 = "mhlo.batch_norm_inference"(%818, %521, %520, %519, %518) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %820 = mhlo.maximum %819, %325 : tensor<1x14x14x256xf32>
  %821 = "mhlo.convolution"(%820, %529) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %822 = "mhlo.broadcast_in_dim"(%528) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %823 = mhlo.add %821, %822 : tensor<1x14x14x256xf32>
  %824 = "mhlo.batch_norm_inference"(%823, %527, %526, %525, %524) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %825 = mhlo.maximum %824, %325 : tensor<1x14x14x256xf32>
  %826 = "mhlo.convolution"(%825, %535) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %827 = "mhlo.broadcast_in_dim"(%534) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %828 = mhlo.add %826, %827 : tensor<1x14x14x1024xf32>
  %829 = "mhlo.batch_norm_inference"(%828, %533, %532, %531, %530) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %830 = mhlo.add %815, %829 : tensor<1x14x14x1024xf32>
  %831 = mhlo.maximum %830, %326 : tensor<1x14x14x1024xf32>
  %832 = "mhlo.convolution"(%831, %541) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
  %833 = "mhlo.broadcast_in_dim"(%540) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %834 = mhlo.add %832, %833 : tensor<1x14x14x256xf32>
  %835 = "mhlo.batch_norm_inference"(%834, %539, %538, %537, %536) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %836 = mhlo.maximum %835, %325 : tensor<1x14x14x256xf32>
  %837 = "mhlo.convolution"(%836, %547) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %838 = "mhlo.broadcast_in_dim"(%546) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %839 = mhlo.add %837, %838 : tensor<1x14x14x256xf32>
  %840 = "mhlo.batch_norm_inference"(%839, %545, %544, %543, %542) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %841 = mhlo.maximum %840, %325 : tensor<1x14x14x256xf32>
  %842 = "mhlo.convolution"(%841, %553) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %843 = "mhlo.broadcast_in_dim"(%552) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %844 = mhlo.add %842, %843 : tensor<1x14x14x1024xf32>
  %845 = "mhlo.batch_norm_inference"(%844, %551, %550, %549, %548) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %846 = mhlo.add %831, %845 : tensor<1x14x14x1024xf32>
  %847 = mhlo.maximum %846, %326 : tensor<1x14x14x1024xf32>
  %848 = "mhlo.convolution"(%847, %559) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
  %849 = "mhlo.broadcast_in_dim"(%558) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %850 = mhlo.add %848, %849 : tensor<1x14x14x256xf32>
  %851 = "mhlo.batch_norm_inference"(%850, %557, %556, %555, %554) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %852 = mhlo.maximum %851, %325 : tensor<1x14x14x256xf32>
  %853 = "mhlo.convolution"(%852, %565) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %854 = "mhlo.broadcast_in_dim"(%564) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %855 = mhlo.add %853, %854 : tensor<1x14x14x256xf32>
  %856 = "mhlo.batch_norm_inference"(%855, %563, %562, %561, %560) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %857 = mhlo.maximum %856, %325 : tensor<1x14x14x256xf32>
  %858 = "mhlo.convolution"(%857, %571) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %859 = "mhlo.broadcast_in_dim"(%570) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %860 = mhlo.add %858, %859 : tensor<1x14x14x1024xf32>
  %861 = "mhlo.batch_norm_inference"(%860, %569, %568, %567, %566) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %862 = mhlo.add %847, %861 : tensor<1x14x14x1024xf32>
  %863 = mhlo.maximum %862, %326 : tensor<1x14x14x1024xf32>
  %864 = "mhlo.convolution"(%863, %577) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x256xf32>) -> tensor<1x14x14x256xf32>
  %865 = "mhlo.broadcast_in_dim"(%576) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %866 = mhlo.add %864, %865 : tensor<1x14x14x256xf32>
  %867 = "mhlo.batch_norm_inference"(%866, %575, %574, %573, %572) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %868 = mhlo.maximum %867, %325 : tensor<1x14x14x256xf32>
  %869 = "mhlo.convolution"(%868, %583) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<3x3x256x256xf32>) -> tensor<1x14x14x256xf32>
  %870 = "mhlo.broadcast_in_dim"(%582) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %871 = mhlo.add %869, %870 : tensor<1x14x14x256xf32>
  %872 = "mhlo.batch_norm_inference"(%871, %581, %580, %579, %578) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x14x14x256xf32>
  %873 = mhlo.maximum %872, %325 : tensor<1x14x14x256xf32>
  %874 = "mhlo.convolution"(%873, %589) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x256xf32>, tensor<1x1x256x1024xf32>) -> tensor<1x14x14x1024xf32>
  %875 = "mhlo.broadcast_in_dim"(%588) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %876 = mhlo.add %874, %875 : tensor<1x14x14x1024xf32>
  %877 = "mhlo.batch_norm_inference"(%876, %587, %586, %585, %584) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x14x14x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x14x14x1024xf32>
  %878 = mhlo.add %863, %877 : tensor<1x14x14x1024xf32>
  %879 = mhlo.maximum %878, %326 : tensor<1x14x14x1024xf32>
  %880 = "mhlo.convolution"(%879, %595) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x2048xf32>) -> tensor<1x7x7x2048xf32>
  %881 = "mhlo.broadcast_in_dim"(%594) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %882 = mhlo.add %880, %881 : tensor<1x7x7x2048xf32>
  %883 = "mhlo.batch_norm_inference"(%882, %593, %592, %591, %590) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %884 = "mhlo.convolution"(%879, %601) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x14x14x1024xf32>, tensor<1x1x1024x512xf32>) -> tensor<1x7x7x512xf32>
  %885 = "mhlo.broadcast_in_dim"(%600) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %886 = mhlo.add %884, %885 : tensor<1x7x7x512xf32>
  %887 = "mhlo.batch_norm_inference"(%886, %599, %598, %597, %596) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %888 = mhlo.maximum %887, %327 : tensor<1x7x7x512xf32>
  %889 = "mhlo.convolution"(%888, %607) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
  %890 = "mhlo.broadcast_in_dim"(%606) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %891 = mhlo.add %889, %890 : tensor<1x7x7x512xf32>
  %892 = "mhlo.batch_norm_inference"(%891, %605, %604, %603, %602) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %893 = mhlo.maximum %892, %327 : tensor<1x7x7x512xf32>
  %894 = "mhlo.convolution"(%893, %613) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
  %895 = "mhlo.broadcast_in_dim"(%612) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %896 = mhlo.add %894, %895 : tensor<1x7x7x2048xf32>
  %897 = "mhlo.batch_norm_inference"(%896, %611, %610, %609, %608) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %898 = mhlo.add %883, %897 : tensor<1x7x7x2048xf32>
  %899 = mhlo.maximum %898, %328 : tensor<1x7x7x2048xf32>
  %900 = "mhlo.convolution"(%899, %619) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
  %901 = "mhlo.broadcast_in_dim"(%618) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %902 = mhlo.add %900, %901 : tensor<1x7x7x512xf32>
  %903 = "mhlo.batch_norm_inference"(%902, %617, %616, %615, %614) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %904 = mhlo.maximum %903, %327 : tensor<1x7x7x512xf32>
  %905 = "mhlo.convolution"(%904, %625) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
  %906 = "mhlo.broadcast_in_dim"(%624) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %907 = mhlo.add %905, %906 : tensor<1x7x7x512xf32>
  %908 = "mhlo.batch_norm_inference"(%907, %623, %622, %621, %620) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %909 = mhlo.maximum %908, %327 : tensor<1x7x7x512xf32>
  %910 = "mhlo.convolution"(%909, %631) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
  %911 = "mhlo.broadcast_in_dim"(%630) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %912 = mhlo.add %910, %911 : tensor<1x7x7x2048xf32>
  %913 = "mhlo.batch_norm_inference"(%912, %629, %628, %627, %626) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %914 = mhlo.add %899, %913 : tensor<1x7x7x2048xf32>
  %915 = mhlo.maximum %914, %328 : tensor<1x7x7x2048xf32>
  %916 = "mhlo.convolution"(%915, %637) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<1x7x7x512xf32>
  %917 = "mhlo.broadcast_in_dim"(%636) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %918 = mhlo.add %916, %917 : tensor<1x7x7x512xf32>
  %919 = "mhlo.batch_norm_inference"(%918, %635, %634, %633, %632) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %920 = mhlo.maximum %919, %327 : tensor<1x7x7x512xf32>
  %921 = "mhlo.convolution"(%920, %643) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<3x3x512x512xf32>) -> tensor<1x7x7x512xf32>
  %922 = "mhlo.broadcast_in_dim"(%642) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %923 = mhlo.add %921, %922 : tensor<1x7x7x512xf32>
  %924 = "mhlo.batch_norm_inference"(%923, %641, %640, %639, %638) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x7x7x512xf32>
  %925 = mhlo.maximum %924, %327 : tensor<1x7x7x512xf32>
  %926 = "mhlo.convolution"(%925, %649) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x512xf32>, tensor<1x1x512x2048xf32>) -> tensor<1x7x7x2048xf32>
  %927 = "mhlo.broadcast_in_dim"(%648) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %928 = mhlo.add %926, %927 : tensor<1x7x7x2048xf32>
  %929 = "mhlo.batch_norm_inference"(%928, %647, %646, %645, %644) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x7x7x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x7x7x2048xf32>
  %930 = mhlo.add %915, %929 : tensor<1x7x7x2048xf32>
  %931 = mhlo.maximum %930, %328 : tensor<1x7x7x2048xf32>
  %932 = "mhlo.reduce"(%931, %331) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %944 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%944) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x2048xf32>, tensor<f32>) -> tensor<1x2048xf32>
  %933 = mhlo.divide %932, %329 : tensor<1x2048xf32>
  %934 = "mhlo.dot"(%933, %651) : (tensor<1x2048xf32>, tensor<2048x1000xf32>) -> tensor<1x1000xf32>
  %935 = "mhlo.broadcast_in_dim"(%650) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1000xf32>
  %936 = mhlo.add %934, %935 : tensor<1x1000xf32>
  %937 = "mhlo.reduce"(%936, %330) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %944 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%944) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
  %938 = "mhlo.broadcast_in_dim"(%937) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
  %939 = mhlo.subtract %936, %938 : tensor<1x1000xf32>
  %940 = "mhlo.exponential"(%939) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
  %941 = "mhlo.reduce"(%940, %331) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %944 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%944) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
  %942 = "mhlo.broadcast_in_dim"(%941) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
  %943 = mhlo.divide %940, %942 : tensor<1x1000xf32>
  return %943 : tensor<1x1000xf32>
}
