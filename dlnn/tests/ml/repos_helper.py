import numpy

from dlnn.tests.ml.testcase import TestCase

label_init = numpy.array([1, 1, 1, 1, 2, 2, 3, 3]) - 1
categorical_label_init = numpy.array([
    [1, -1, -1],
    [1, -1, -1],
    [1, -1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
])
normalized = numpy.array([[[[0.84333333, 0.43666667, 0.86333333, 0.31666667],
                            [0.84333333, 0.43666667, 0.86333333, 0.31666667],
                            [0.84333333, 0.43666667, 0.86333333, 0.31666667],
                            [0.84333333, 0.43666667, 0.86333333, 0.31666667]]],
                          [[[0.43666667, 0.86333333, 0.31666667, 0.87666667],
                            [0.43666667, 0.86333333, 0.31666667, 0.87666667],
                            [0.43666667, 0.86333333, 0.31666667, 0.87666667],
                            [0.43666667, 0.86333333, 0.31666667, 0.87666667]]],
                          [[[0.86333333, 0.31666667, 0.87666667, 0.16666667],
                            [0.86333333, 0.31666667, 0.87666667, 0.16666667],
                            [0.86333333, 0.31666667, 0.87666667, 0.16666667],
                            [0.86333333, 0.31666667, 0.87666667, 0.16666667]]],
                          [[[0.31666667, 0.87666667, 0.16666667, 0.51333333],
                            [0.31666667, 0.87666667, 0.16666667, 0.51333333],
                            [0.31666667, 0.87666667, 0.16666667, 0.51333333],
                            [0.31666667, 0.87666667, 0.16666667, 0.51333333]]],
                          [[[0.87666667, 0.16666667, 0.51333333, 0.24333333],
                            [0.87666667, 0.16666667, 0.51333333, 0.24333333],
                            [0.87666667, 0.16666667, 0.51333333, 0.24333333],
                            [0.87666667, 0.16666667, 0.51333333, 0.24333333]]],
                          [[[0.16666667, 0.51333333, 0.24333333, 0.470000000],
                            [0.16666667, 0.51333333, 0.24333333, 0.470000000],
                            [0.16666667, 0.51333333, 0.24333333, 0.470000000],
                            [0.16666667, 0.51333333, 0.24333333, 0.470000000]]],
                          [[[0.51333333, 0.24333333, 0.47000000, 0.486666670],
                            [0.51333333, 0.24333333, 0.47000000, 0.486666670],
                            [0.51333333, 0.24333333, 0.47000000, 0.486666670],
                            [0.51333333, 0.24333333, 0.47000000, 0.486666670]]],
                          [[[0.24333333, 0.47000000, 0.48666667, 0.290000000],
                            [0.24333333, 0.47000000, 0.48666667, 0.290000000],
                            [0.24333333, 0.47000000, 0.48666667, 0.290000000],
                            [0.24333333, 0.47000000, 0.48666667, 0.290000000]]]])
corr_step_1_seg_1 = numpy.array([[0.28444444, 0.47629630, 0.35925926, 0.26222222],
                                 [0.42666667, 0.71444444, 0.53888889, 0.39333333],
                                 [0.42666667, 0.71444444, 0.53888889, 0.39333333],
                                 [0.28444444, 0.47629630, 0.35925926, 0.26222222]])
corr_step_1_seg_2 = numpy.array([[0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333],
                                 [0.84333333, 0.86333333, 0.86333333, 0.86333333]])
corr_step_1_seg_3 = numpy.array([[0.36667424, 0.39571821, 0.33745278, 0.36612839],
                                 [0.36525105, 0.20851325, 0.24881943, 0.37823053],
                                 [0.36525105, 0.20851325, 0.24881943, 0.37823053],
                                 [0.36667424, 0.39571821, 0.33745278, 0.36612839]])
corr_step_1 = numpy.concatenate(([[corr_step_1_seg_1]], [[corr_step_1_seg_2]], [[corr_step_1_seg_3]]), axis=1)
corr_step_2_seg_1 = numpy.array([[0.57063550, 0.61687292, 0.58886111, 0.56518248],
                                 [0.60507742, 0.67138247, 0.63155391, 0.59708487],
                                 [0.60507742, 0.67138247, 0.63155391, 0.59708487],
                                 [0.57063550, 0.61687292, 0.58886111, 0.56518248]])
corr_step_2_seg_2 = numpy.array([[0.69916679, 0.70335661, 0.70335661, 0.70335661],
                                 [0.69916679, 0.70335661, 0.70335661, 0.70335661],
                                 [0.69916679, 0.70335661, 0.70335661, 0.70335661],
                                 [0.69916679, 0.70335661, 0.70335661, 0.70335661]])
corr_step_2_seg_3 = numpy.array([[0.59065511, 0.59765848, 0.58357164, 0.59052313],
                                 [0.59031097, 0.55194026, 0.56188590, 0.59344626],
                                 [0.59031097, 0.55194026, 0.56188590, 0.59344626],
                                 [0.59065511, 0.59765848, 0.58357164, 0.59052313]])
corr_step_2 = numpy.concatenate(([[corr_step_2_seg_1]], [[corr_step_2_seg_2]], [[corr_step_2_seg_3]]), axis=1)
corr_step_3_seg_1 = numpy.array([[0.27377427, 0.40937594, 0.40788200, 0.26474250],
                                 [0.41560313, 0.62137740, 0.61899540, 0.40125790],
                                 [0.41560316, 0.62137747, 0.61899540, 0.40125790],
                                 [0.27377427, 0.40937594, 0.40788198, 0.26474250]])
corr_step_3_seg_2 = numpy.array([[0.70335660, 0.70335660, 0.70335660, 0.70335660],
                                 [0.70335660, 0.70335660, 0.70335660, 0.70335660],
                                 [0.70335660, 0.70335660, 0.70335660, 0.70335660],
                                 [0.70335660, 0.70335660, 0.70335660, 0.70335660]])
corr_step_3_seg_3 = numpy.array([[0.30734155, 0.29002590, 0.29029182, 0.30705318],
                                 [0.28987694, 0.01838925, 0.01904486, 0.29064130],
                                 [0.28987694, 0.01838925, 0.01904486, 0.29064130],
                                 [0.30734155, 0.29002586, 0.29029182, 0.30705318]])
corr_step_3 = numpy.concatenate(([[corr_step_3_seg_1]], [[corr_step_3_seg_2]], [[corr_step_3_seg_3]]), axis=1)
corr_step_4_seg_1 = numpy.array([[0.56801925, 0.60093823, 0.60057991, 0.56580174],
                                 [0.60243064, 0.65053175, 0.64999003, 0.59898985],
                                 [0.60243065, 0.65053177, 0.64999003, 0.59898985],
                                 [0.56801925, 0.60093823, 0.60057991, 0.56580174]])
corr_step_4_seg_2 = numpy.array([[0.66893155, 0.66893155, 0.66893155, 0.66893155],
                                 [0.66893155, 0.66893155, 0.66893155, 0.66893155],
                                 [0.66893155, 0.66893155, 0.66893155, 0.66893155],
                                 [0.66893155, 0.66893155, 0.66893155, 0.66893155]])
corr_step_4_seg_3 = numpy.array([[0.57623623, 0.57200247, 0.57206757, 0.57616581],
                                 [0.57196601, 0.50459718, 0.50476107, 0.57215313],
                                 [0.57196601, 0.50459718, 0.50476107, 0.57215313],
                                 [0.57623623, 0.57200246, 0.57206757, 0.57616581]])
corr_step_4 = numpy.concatenate(([[corr_step_4_seg_1]], [[corr_step_4_seg_2]], [[corr_step_4_seg_3]]), axis=1)
corr_step_5_seg_1 = numpy.array([[0.65053177, 0.64999000],
                                 [0.65053177, 0.64999000]])
corr_step_5_seg_2 = numpy.array([[0.66893154, 0.66893154],
                                 [0.66893154, 0.66893154]])
corr_step_5_seg_3 = numpy.array([[0.57623625, 0.57616580],
                                 [0.57623625, 0.57616580]])
corr_step_5 = numpy.concatenate(([[corr_step_5_seg_1]], [[corr_step_5_seg_2]], [[corr_step_5_seg_3]]), axis=1)
corr_step_6_seg_1 = numpy.array([[0.28900486, 0.28900486],
                                 [0.28900486, 0.28900486]])
corr_step_6_seg_2 = numpy.array([[0.66893154, 0.66893154],
                                 [0.66893154, 0.66893154]])
corr_step_6_seg_3 = numpy.array([[0.30368462, 0.30368460],
                                 [0.30368460, 0.30368460]])
corr_step_6 = numpy.concatenate(([[corr_step_6_seg_1]], [[corr_step_6_seg_2]], [[corr_step_6_seg_3]]), axis=1)
corr_step_7_seg_1 = numpy.array([[0.57175249, 0.57175249],
                                 [0.57175249, 0.57175249]])
corr_step_7_seg_2 = numpy.array([[0.66126387, 0.66126387],
                                 [0.66126387, 0.66126387]])
corr_step_7_seg_3 = numpy.array([[0.57534300, 0.57534300],
                                 [0.57534300, 0.57534300]])
corr_step_7 = numpy.concatenate(([[corr_step_7_seg_1]], [[corr_step_7_seg_2]], [[corr_step_7_seg_3]]), axis=1)
corr_step_8_seg_1 = numpy.array([[0.57175250, 0.57175250],
                                 [0.57175250, 0.57175250]])
corr_step_8_seg_2 = numpy.array([[0.66126390, 0.66126390],
                                 [0.66126390, 0.66126390]])
corr_step_8_seg_3 = numpy.array([[0.57534300, 0.57534300],
                                 [0.57534300, 0.57534300]])
corr_step_8 = numpy.concatenate(([[corr_step_8_seg_1]], [[corr_step_8_seg_2]], [[corr_step_8_seg_3]]), axis=1)
corr_step_8_full = numpy.array([[[[0.57175250, 0.57175250],
                                  [0.57175250, 0.57175250]],
                                 [[0.66126390, 0.66126390],
                                  [0.66126390, 0.66126390]],
                                 [[0.57534300, 0.57534300],
                                  [0.57534300, 0.57534300]]],
                                [[[0.57171450, 0.57171450],
                                  [0.57171450, 0.57171450]],
                                 [[0.66140145, 0.66140145],
                                  [0.66140145, 0.66140145]],
                                 [[0.57539700, 0.57539700],
                                  [0.57539700, 0.57539700]]],
                                [[[0.57148540, 0.57148540],
                                  [0.57148540, 0.57148540]],
                                 [[0.66140145, 0.66140145],
                                  [0.66140145, 0.66140145]],
                                 [[0.57544700, 0.57544700],
                                  [0.57544700, 0.57544700]]],
                                [[[0.57109510, 0.57109510],
                                  [0.57109510, 0.57109510]],
                                 [[0.66140145, 0.66140145],
                                  [0.66140145, 0.66140145]],
                                 [[0.57527700, 0.57527700],
                                  [0.57527700, 0.57527700]]],
                                [[[0.57081240, 0.57081240],
                                  [0.57081240, 0.57081240]],
                                 [[0.66140145, 0.66140145],
                                  [0.66140145, 0.66140145]],
                                 [[0.57512546, 0.57512546],
                                  [0.57512546, 0.57512546]]],
                                [[[0.57049360, 0.57049360],
                                  [0.57049360, 0.57049360]],
                                 [[0.65734580, 0.65734580],
                                  [0.65734580, 0.65734580]],
                                 [[0.57477814, 0.57477814],
                                  [0.57477814, 0.57477814]]],
                                [[[0.57075800, 0.57075800],
                                  [0.57075800, 0.57075800]],
                                 [[0.65734580, 0.65734580],
                                  [0.65734580, 0.65734580]],
                                 [[0.57481300, 0.57481300],
                                  [0.57481300, 0.57481300]]],
                                [[[0.57070196, 0.57070196],
                                  [0.57070196, 0.57070196]],
                                 [[0.65702490, 0.65702490],
                                  [0.65702490, 0.65702490]],
                                 [[0.57474244, 0.57474244],
                                  [0.57474244, 0.57474244]]]])
corr_step_9 = numpy.array([[0.57175250, 0.57175250, 0.57175250, 0.57175250,
                            0.66126390, 0.66126390, 0.66126390, 0.66126390,
                            0.57534300, 0.57534300, 0.57534300, 0.57534300],
                           [0.57171450, 0.57171450, 0.57171450, 0.57171450,
                            0.66140145, 0.66140145, 0.66140145, 0.66140145,
                            0.57539700, 0.57539700, 0.57539700, 0.57539700],
                           [0.57148540, 0.57148540, 0.57148540, 0.57148540,
                            0.66140145, 0.66140145, 0.66140145, 0.66140145,
                            0.57544700, 0.57544700, 0.57544700, 0.57544700],
                           [0.57109510, 0.57109510, 0.57109510, 0.57109510,
                            0.66140145, 0.66140145, 0.66140145, 0.66140145,
                            0.57527700, 0.57527700, 0.57527700, 0.57527700],
                           [0.57081240, 0.57081240, 0.57081240, 0.57081240,
                            0.66140145, 0.66140145, 0.66140145, 0.66140145,
                            0.57512546, 0.57512546, 0.57512546, 0.57512546],
                           [0.57049360, 0.57049360, 0.57049360, 0.57049360,
                            0.65734580, 0.65734580, 0.65734580, 0.65734580,
                            0.57477814, 0.57477814, 0.57477814, 0.57477814],
                           [0.57075800, 0.57075800, 0.57075800, 0.57075800,
                            0.65734580, 0.65734580, 0.65734580, 0.65734580,
                            0.57481300, 0.57481300, 0.57481300, 0.57481300],
                           [0.57070196, 0.57070196, 0.57070196, 0.57070196,
                            0.65702490, 0.65702490, 0.65702490, 0.65702490,
                            0.57474244, 0.57474244, 0.57474244, 0.57474244]])
corr_step_10_a_dummy_kernel_init = numpy.array([[+0.03040000, +0.34330000, -0.28470000, +0.21670000, +0.36270000],
                                                [-0.02870000, -0.03950000, +0.44250000, +0.12830000, +0.30640000],
                                                [-0.09850000, -0.17860000, -0.12120000, +0.21900000, -0.30690000],
                                                [-0.01000000, -0.37920000, -0.36010000, +0.44720000, +0.13090000],
                                                [-0.06130000, +0.20780000, +0.02960000, +0.15240000, +0.27820000],
                                                [+0.11800000, +0.00210000, -0.01460000, -0.38990000, +0.41650000],
                                                [-0.06740000, -0.07300000, -0.47070000, +0.01040000, +0.46370000],
                                                [-0.22460000, +0.24340000, -0.38320000, +0.26930000, +0.26570000],
                                                [+0.04080000, -0.33070000, +0.43010000, -0.29180000, -0.48020000],
                                                [+0.10160000, +0.10740000, -0.03110000, +0.09630000, +0.22350000],
                                                [-0.09190000, +0.09760000, -0.07430000, -0.37820000, -0.15000000],
                                                [+0.04670000, -0.37290000, -0.41850000, -0.21050000, +0.17680000]])
corr_step_10_a_dummy_bias_init = numpy.array([0.71943, 0.65004, 0.72691, 0.37385, 0.58158])
corr_step_10_a_dummy_non_bias = numpy.array([[-0.16073523, -0.18061252, -0.79366340, 0.15487751, 1.09136570],
                                             [-0.16075826, -0.18057747, -0.79377160, 0.15480250, 1.09153040],
                                             [-0.16072893, -0.18054423, -0.79370220, 0.15453164, 1.09140600],
                                             [-0.16070378, -0.18036029, -0.79355990, 0.15427034, 1.09125260],
                                             [-0.16068833, -0.18021297, -0.79345423, 0.15410328, 1.09114810],
                                             [-0.15973374, -0.18150117, -0.78991630, 0.15388210, 1.08529510],
                                             [-0.15975860, -0.18158573, -0.79000500, 0.15412208, 1.08541740],
                                             [-0.15968394, -0.18165833, -0.78971110, 0.15410726, 1.08494900]])
corr_step_10_a_dummy = numpy.array([[+0.55869480, +0.46942747, -0.06675339, +0.52872753, +1.67294570],
                                    [+0.55867180, +0.46946250, -0.06686163, +0.52865250, +1.67311050],
                                    [+0.55870110, +0.46949574, -0.06679219, +0.52838165, +1.67298600],
                                    [+0.55872625, +0.46967968, -0.06664991, +0.52812034, +1.67283250],
                                    [+0.55874170, +0.46982700, -0.06654423, +0.52795327, +1.67272810],
                                    [+0.55969630, +0.46853882, -0.06300628, +0.52773210, +1.66687510],
                                    [+0.55967140, +0.46845424, -0.06309503, +0.52797210, +1.66699740],
                                    [+0.55974610, +0.46838164, -0.06280112, +0.52795726, +1.66652890]])
corr_step_11_a_dummy = numpy.array([[0.63615050, 0.61524820, 0.48331788, 0.62918630, 0.84196820],
                                    [0.63614510, 0.61525655, 0.48329080, 0.62916880, 0.84199005],
                                    [0.63615190, 0.61526436, 0.48330817, 0.62910557, 0.84197360],
                                    [0.63615775, 0.61530800, 0.48334372, 0.62904460, 0.84195310],
                                    [0.63616130, 0.61534280, 0.48337013, 0.62900570, 0.84193920],
                                    [0.63638230, 0.61503786, 0.48425367, 0.62895400, 0.84115875],
                                    [0.63637650, 0.61501783, 0.48423147, 0.62900996, 0.84117514],
                                    [0.63639380, 0.61500060, 0.48430488, 0.62900656, 0.84111243]])
corr_step_10_b_dummy_kernel_init = numpy.array(
    [[+0.38442000, +0.21265000, -0.12185000, -0.25108000, -0.24715000, +0.26724000, -0.45014000],
     [+0.18529000, +0.12028000, +0.24668000, +0.47726000, -0.11609000, -0.23979000, +0.37747000],
     [+0.30610000, -0.03887900, -0.40904000, +0.06426900, -0.31262000, +0.03169000, -0.14497000],
     [-0.18522000, +0.22674000, +0.01577300, +0.29064000, -0.29551000, +0.17811000, -0.44751000],
     [+0.30117000, +0.17857000, +0.44601000, -0.40844000, +0.40844000, +0.00995300, +0.11490000],
     [-0.18393000, -0.42251000, +0.35061000, -0.35547000, -0.12951000, +0.12239000, +0.49755000],
     [+0.01734400, +0.49051000, -0.27347000, -0.10199000, +0.19657000, -0.43536000, +0.24766000],
     [-0.07960000, +0.31132000, -0.12039000, -0.18093000, +0.48605000, +0.21818000, -0.08681700],
     [-0.40137000, +0.23456000, +0.13731000, -0.42616000, -0.37949000, +0.48160000, -0.00320060],
     [-0.47759000, -0.44617000, -0.35913000, +0.39347000, -0.03418000, +0.06085700, -0.00554370],
     [-0.43221000, +0.39765000, -0.21143000, -0.23095000, +0.09419400, -0.02412100, -0.13169000],
     [+0.15561000, +0.43820000, +0.12043000, -0.21716000, -0.29482000, -0.06086600, -0.47275000]]
)
corr_step_10_b_dummy_bias_init = numpy.array(
    [0.87618000, 0.61009000, 0.20359000, 0.51992000, 0.05382400, 0.86219000, 0.44293293])
corr_step_10_b_dummy_non_bias = numpy.array(
    [[-0.23363790, +1.02582810, -0.06712769, -0.63661670, -0.27297583, +0.34275080, -0.22174174],
     [-0.23371892, +1.02591880, -0.06707902, -0.63680875, -0.27283987, +0.34275480, -0.22164320],
     [-0.23393495, +1.02583070, -0.06703314, -0.63696593, -0.27264804, +0.34272330, -0.22152142],
     [-0.23400797, +1.02552130, -0.06687519, -0.63711100, -0.27216443, +0.34255293, -0.22115757],
     [-0.23402819, +1.02527950, -0.06675191, -0.63720244, -0.27179676, +0.34241652, -0.22087663],
     [-0.23406997, +1.02263400, -0.06819114, -0.63297510, -0.27517343, +0.34252608, -0.22358789],
     [-0.23392768, +1.02279350, -0.06827298, -0.63283825, -0.27545172, +0.34260476, -0.22378509],
     [-0.23390248, +1.02254130, -0.06836513, -0.63250095, -0.27566242, +0.34258643, -0.22395270]]
)
corr_step_10_b_dummy = numpy.array(
    [[+0.64254210, +1.63591810, +0.13646232, -0.11669672, -0.21915182, +1.20494080, +0.22119120],
     [+0.64246106, +1.63600890, +0.13651098, -0.11688876, -0.21901587, +1.20494480, +0.22128974],
     [+0.64224505, +1.63592080, +0.13655686, -0.11704594, -0.21882403, +1.20491340, +0.22141151],
     [+0.64217204, +1.63561130, +0.13671482, -0.11719102, -0.21834043, +1.20474290, +0.22177537],
     [+0.64215183, +1.63536950, +0.13683810, -0.11728245, -0.21797276, +1.20460650, +0.22205630],
     [+0.64211000, +1.63272400, +0.13539886, -0.11305511, -0.22134942, +1.20471610, +0.21934505],
     [+0.64225230, +1.63288350, +0.13531703, -0.11291826, -0.22162771, +1.20479480, +0.21914785],
     [+0.64227750, +1.63263130, +0.13522488, -0.11258096, -0.22183841, +1.20477640, +0.21898024]]
)


class ReposHelper(TestCase):
    def test_normalize(self):
        self.assertIsNotNone(normalized)
        # print(normalized)

    def test_corr_step_1_seg_1(self):
        self.assertIsNotNone(corr_step_1_seg_1)
        # print(corr_step_1_seg_1)

    def test_corr_step_1_seg_2(self):
        self.assertIsNotNone(corr_step_1_seg_2)
        # print(corr_step_1_seg_2)

    def test_corr_step_1_seg_3(self):
        self.assertIsNotNone(corr_step_1_seg_3)
        # print(corr_step_1_seg_3)

    def test_corr_step_1(self):
        self.assertIsNotNone(corr_step_1)
        # print(corr_step_1)
