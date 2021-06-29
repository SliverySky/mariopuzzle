initial_states = [
[-0.2993487289,0.2554769395,0.9707999524,-0.1047808205,0.4402980404,0.8715019636,0.6046720138,0.2090476390,-0.1229728329,0.6307970805,0.9025042194,0.7144022808,0.1602081151,0.2734302014,0.0868468465,0.6525413950,-0.2802780360,0.3750714419,-0.6626598890,-0.0207016341,0.4063147180,-0.6559981280,-0.4404730063,-0.9997227581,-0.5893812004,0.3761141740,0.8639538503,-0.8551728413,0.2385622734,-0.5514413382,0.4683871379,0.0643224886],
[-0.1463104723,-0.0352590797,-0.9689904382,0.1998383874,-0.9765401754,-0.2705418575,-0.1270299049,-0.0673802036,-0.1533766485,-0.8330940719,-0.9591187724,0.2948880735,0.6234275468,0.8628111160,0.5636378342,0.4851442225,-0.4911779773,0.2884322024,-0.4874767677,0.9667711511,-0.4516670012,-0.1127399471,-0.7269874652,-0.4629463793,0.5012852942,0.0315969592,-0.4392210158,0.7542190664,-0.0960478980,0.4769558463,-0.7287853265,0.9241763767],
[-0.9990824891,0.1534014449,0.9204917879,-0.2553941245,0.8388112640,-0.9875997848,0.1500143829,-0.3507938898,0.9441494821,-0.5873940535,0.3180151200,-0.2176778306,0.2217988795,-0.6196645128,-0.0065920052,0.1144761349,-0.9864552436,-0.8900894637,-0.5697906184,0.8516729112,-0.0451211633,0.1231657353,0.5701183181,-0.9403275683,0.8802708727,-0.1636622809,-0.0945023824,-0.1505494022,0.7085879801,-0.6259212861,-0.4905710142,-0.2973407270],
[-0.3834142746,0.1704755597,-0.4634424331,0.7308188544,0.5345524792,-0.1769318201,-0.2948849896,-0.7052768100,0.4258714161,-0.3079517060,-0.2377096231,-0.4820130459,0.6420712060,0.1998345089,-0.0740021672,-0.0722915641,-0.0902497057,0.2578084785,0.8325162531,0.3466768652,0.8572446227,0.5019982953,-0.6228410019,0.5853337467,0.4112619578,-0.8857032522,-0.6559408237,-0.9617244670,-0.3877135173,0.6771725162,-0.7405654730,-0.0590221999],
[-0.0117603544,0.5435911411,-0.9194058191,-0.1677490791,0.8704073161,0.4133406463,0.7855591055,-0.8007854486,-0.2519281134,0.1143233159,0.3268991082,0.2290696703,-0.7050846201,0.2085122019,-0.5438507958,-0.3677548400,-0.4848287972,-0.3557720098,-0.9548053064,0.8555380929,-0.4139845426,0.6852190340,-0.8815384727,0.6424847799,0.9417498767,-0.2605143855,0.9268129804,-0.4259264806,0.6026560454,-0.2951741605,-0.6606589277,0.0746261220],
[-0.9870994214,0.3662382825,-0.5823306038,-0.4851250169,0.4691742956,0.9908169476,-0.7716083730,0.9449284634,0.0607104564,0.5051160778,0.4971608397,0.0760813342,0.5022104409,-0.7221088215,-0.7271978246,0.2812225858,0.8925786372,-0.9716389358,0.8714364061,0.3697431824,-0.2910461485,-0.8806972958,-0.8193889792,0.0324030519,-0.9518877117,0.0610817263,0.0526827528,-0.0642484422,-0.9465640459,0.7985227049,-0.4786504296,-0.7823036248] ,
[-0.9082752812,0.6704337441,-0.8807566345,0.4943016006,-0.9585995062,0.0552510981,0.4557974245,-0.6577187147,-0.9610603737,0.9396825512,-0.3325694542,-0.4129183976,0.9809991017,0.5197039074,0.8076763786,-0.8589100219,0.3196943699,-0.4990423646,0.2435179931,0.2936383753,0.7662399338,0.4842398556,-0.2300217777,-0.6578546122,-0.4483851830,0.9754991803,0.8014557695,-0.4050986071,-0.4213803737,-0.5831317190,0.8631491606,-0.7319316497],
[0.4453462162,0.9597547869,-0.5860881314,0.0761462622,-0.9896541507,0.1505945833,-0.6810731410,0.8381536710,-0.9495195510,-0.6002985433,0.5513431479,0.7290309310,0.3554182454,0.0228187921,-0.6920741686,-0.4292558165,-0.8809383529,0.7871316154,0.9273614440,-0.4474431919,0.1401999432,-0.9429184711,-0.8659299717,-0.0559972849,0.6539272680,-0.6370024824,-0.9275892172,-0.3721340340,-0.0182234657,0.5083880305,-0.0907019382,0.4148421280] ,
[-0.1579650777,-0.2591214892,-0.5449687890,-0.2303608466,0.3765891034,-0.1735907304,-0.7089829476,-0.1971666695,0.1627500692,0.6624881950,-0.0785501378,-0.7091834780,-0.4412309180,0.9859693395,0.9459553814,0.6373656824,-0.0168159774,0.5030617350,0.0643132150,-0.8713398271,-0.7804082665,-0.2078522321,0.2691452665,0.3326009667,-0.3847122341,-0.0596632018,0.5617064637,0.1351068058,-0.9150242587,0.5118733003,0.0056682277,-0.3574213063],
[-0.2507230722,-0.7820462316,0.5066522481,-0.9887047110,0.8232544205,-0.8680633451,0.6931364347,-0.1635781384,0.8194063331,0.7732785254,0.0067529555,0.3125079570,-0.8041217016,-0.0129017353,0.2908493604,0.0177573517,0.0041288760,0.1161548717,0.2094496234,0.6097958475,0.2976037441,-0.2751487286,0.4720883858,-0.8409750774,0.1095659657,0.7657412226,0.6072481260,0.7642934648,-0.3824264002,-0.3276767482,0.6100571951,0.8538078703] ,
[0.0424376552,1.0000000000,-1.0000000000,1.0000000000,0.7017688781,-0.3168489152,-0.7098702093,-0.3223290526,-0.3905752925,-1.0000000000,0.0794654813,-0.6871093367,-0.4287814825,0.9123291036,0.5909935352,-1.0000000000,0.1941640796,1.0000000000,0.8880105880,-0.3401264534,0.4826885185,1.0000000000,0.3830340081,-0.2381784984,1.0000000000,0.0274600187,1.0000000000,-1.0000000000,-1.0000000000,1.0000000000,-0.7408831423,-0.4580546690] ,
[0.2685300610,-0.6402481171,-1.0000000000,-0.8171515454,0.1823617622,1.0000000000,0.2556731861,1.0000000000,0.7580679777,-0.2114311712,-1.0000000000,-0.3257054334,-0.3436325201,0.6078988977,-0.7341648089,-0.2025140062,0.3815837553,-1.0000000000,1.0000000000,-1.0000000000,0.4215596055,-1.0000000000,0.3070448611,1.0000000000,-0.1505472375,1.0000000000,1.0000000000,0.5495065776,0.8555022250,1.0000000000,0.5812657417,-0.3287885106] ,
[-0.7278776076,-0.0855717083,0.4366970549,-0.7400810003,-0.4003002968,-0.1503705393,-0.1792333664,0.2467507236,0.0052588242,-1.0000000000,1.0000000000,-1.0000000000,-1.0000000000,-0.0927660104,0.5444713770,0.1839426399,-1.0000000000,-0.0142821386,0.3954378676,1.0000000000,0.9044768420,-0.1583562568,-0.2593840398,-0.8561527079,0.6350609759,-0.9455547998,-0.3225953330,-0.3756073410,1.0000000000,0.0241266746,0.1812731868,0.1566752768] ,
[0.3362176772,-1.0000000000,-0.1232160291,-0.4856837024,0.8212132583,0.2409185926,-1.0000000000,0.6813432192,-0.6509815085,-0.0164713301,-0.3052843092,1.0000000000,-1.0000000000,1.0000000000,-1.0000000000,-0.5703416020,0.7233745532,1.0000000000,-0.6868352635,-0.7697450486,0.7939899525,0.1042479232,1.0000000000,-0.9347771668,0.4594903556,-1.0000000000,-1.0000000000,-0.3735332572,0.9649828027,-1.0000000000,0.3315991114,0.1522362479] ,
[-1.0000000000,-0.9232221250,-0.3560560197,0.3204637914,1.0000000000,1.0000000000,-0.0454115184,-0.5011123291,1.0000000000,0.7297965321,0.7756864067,-0.2565621278,1.0000000000,0.8594947382,-1.0000000000,-0.6405898046,-0.9623711153,-0.5909165878,-0.1012715992,0.5087507858,1.0000000000,0.1252225349,1.0000000000,-1.0000000000,-0.0391594552,1.0000000000,0.7965423527,0.0994177177,0.3528685829,-0.1840453742,1.0000000000,0.5667731628] ,
[0.0174465580,-0.4668142391,1.0000000000,0.0773402014,-0.8196049409,-1.0000000000,0.3921756860,-1.0000000000,-1.0000000000,0.2056031289,0.4556411053,-0.3285529778,-0.1687720119,-0.5284709417,0.4957233623,0.1519850401,-0.9185943153,-0.8031359149,-0.5447967931,-1.0000000000,0.4545632091,0.4317846712,0.1009795330,-0.5991792578,1.0000000000,-0.6696792582,-0.9984144291,-0.8776233385,-0.2786217794,-1.0000000000,-1.0000000000,1.0000000000] ,
[-0.9024332444,0.4970084950,0.5719321295,0.2511248133,0.6655457853,0.0034550136,0.3802479211,1.0000000000,0.6750757012,-1.0000000000,0.5199822591,-0.5140117679,0.2157021871,-1.0000000000,0.1753460054,0.2904737786,0.1429861364,1.0000000000,0.9212911921,1.0000000000,0.3558438921,0.8200365621,-1.0000000000,0.7236487912,1.0000000000,0.0966499017,-0.0280943377,1.0000000000,0.4308511740,-1.0000000000,-0.3468146597,-0.1830212136] ,
[1.0000000000,-0.0043629913,0.8272895753,-0.6999948492,-0.1031577714,0.3179512613,-0.2106739307,0.5098993968,-0.8022392793,0.7251200027,0.7513176572,-0.6384481282,0.2092066961,-0.4086472113,-0.5897648304,-0.1988050416,0.4821432054,0.1208883215,-0.6990448044,-1.0000000000,-0.1331980275,1.0000000000,0.3431456149,0.4034286900,-0.6169556225,0.7442523986,-0.6418658643,-1.0000000000,0.0310401729,-1.0000000000,1.0000000000,-0.2556297343] ,
[0.7042486884,-0.0725846611,0.1646670904,-0.9340917274,-0.0187584407,0.2986565684,0.5769883812,-1.0000000000,-0.7431640679,-1.0000000000,0.7107687283,-1.0000000000,0.8732382604,-0.5366779903,-0.9427505577,1.0000000000,0.3192680884,0.2223124806,0.0866619865,0.6710181161,-1.0000000000,-0.3246064429,-1.0000000000,-0.9617390633,-0.4603416944,-0.7579753477,1.0000000000,-0.5327054504,-0.8391019580,-0.7375838322,-1.0000000000,-1.0000000000] ,
[-1.0000000000,0.4166120681,0.5055813966,1.0000000000,0.1433693817,0.8132255609,-0.1790221722,0.8637824333,0.1777968460,-0.7312435763,1.0000000000,-1.0000000000,0.5428156488,-0.3050959151,-0.0074348001,-0.1056497352,1.0000000000,-1.0000000000,0.3674959092,1.0000000000,-1.0000000000,-0.5368705892,-1.0000000000,1.0000000000,0.7707458844,-1.0000000000,1.0000000000,0.2704084929,0.6505394562,-1.0000000000,-1.0000000000,0.1074269948], 

[-0.6286193268,0.4126233810,1.0000000000,-0.0317954650,-0.7755376153,0.5125192063,0.4100783995,0.1658842206,0.9020305298,-0.1458716589,0.3690924152,-0.4880108026,-0.7051268855,0.5864699318,-0.5085680223,-0.2844350115,-0.8313774123,1.0000000000,0.5781776691,0.7231869099,0.5821220771,-1.0000000000,-1.0000000000,0.7856817046,-0.9439408261,0.0841388993,0.2330205974,-0.5426648826,-1.0000000000,-0.7276826218,-1.0000000000,0.3606131363] ,
[-1.0000000000,-1.0000000000,1.0000000000,-0.7257301392,0.9952698543,0.6878868989,1.0000000000,0.2584160087,-0.6111575373,1.0000000000,-0.8159324991,0.0503344362,0.3102000121,0.0731869900,1.0000000000,0.9993669080,-0.3652452170,0.4721120921,0.0916693950,1.0000000000,-1.0000000000,0.1380789834,-1.0000000000,-0.1481150197,0.5646069960,-0.5596745250,-0.0381617951,0.2099968259,1.0000000000,-0.1789188005,-0.0930014794,0.5705463981] ,
[-0.9967855713,-0.2377177931,1.0000000000,1.0000000000,-1.0000000000,0.0578801208,-1.0000000000,-0.0474540671,0.5126692109,-0.5458003843,-0.4490354823,-0.0844101113,-0.1096831483,1.0000000000,-0.3575750355,1.0000000000,0.2195699973,-0.5524980442,0.8405915909,-1.0000000000,-0.6851966623,-0.5412980346,-0.3588746296,0.1383436966,1.0000000000,-1.0000000000,-0.3602353576,-1.0000000000,-0.6326360874,0.1283238976,-1.0000000000,-1.0000000000] ,
[1.0000000000,-0.5399877965,-1.0000000000,-0.1753796671,-0.4450854417,-1.0000000000,1.0000000000,-0.7476369982,0.1359233594,-0.0088449390,-1.0000000000,-0.4409302857,0.0568501198,1.0000000000,-0.8297985958,-0.6816510782,-1.0000000000,1.0000000000,-0.1848285343,-0.1844936837,0.6670873275,-0.4403568228,0.3269290786,0.0608318091,1.0000000000,-0.3227450354,0.0238674097,-1.0000000000,-0.0916149667,1.0000000000,0.6023575489,0.8736341321] ,
[0.9808353143,1.0000000000,0.1442891861,-1.0000000000,0.8014882119,-1.0000000000,0.0608383770,0.3758951785,0.5150233044,0.4919721920,-0.8130351051,-0.1747349865,0.5583461093,1.0000000000,-0.3731352458,-1.0000000000,0.2885676838,-0.6297561712,0.0025352142,-0.2529874835,0.6004803621,0.4373344009,-1.0000000000,-1.0000000000,0.4347924467,0.4844937107,1.0000000000,-0.8633018479,-1.0000000000,-0.0819646446,1.0000000000,-0.5393797330] ,
[-1.0000000000,0.8447757244,0.0295659619,1.0000000000,0.7972217777,0.0687375698,-0.0748473895,0.7919699245,-0.5648181939,1.0000000000,0.1723454143,0.7461225718,-0.6058909927,0.3560018773,0.1785916354,-0.5878403299,0.0224441670,1.0000000000,-1.0000000000,-0.7500523233,0.4168939844,1.0000000000,0.5460788990,1.0000000000,-1.0000000000,0.6532532876,-0.7305851466,0.4370621041,-1.0000000000,-0.4925672352,-1.0000000000,1.0000000000] ,
[-0.3410173769,-0.9503993972,1.0000000000,-0.1049036720,-0.6775815415,-0.1636719749,-0.5091413236,-0.5892692425,-0.4256675125,0.1416849211,1.0000000000,-0.4802883643,-0.6739368422,-0.5612512271,-0.6152329137,0.9049005403,-0.0273458917,0.7117331893,-0.6846428485,1.0000000000,0.4498078337,0.9885827054,-1.0000000000,0.7958607132,-1.0000000000,0.1380435609,-1.0000000000,-1.0000000000,0.2691630171,1.0000000000,0.0036412779,-0.3313728281] ,
[-1.0000000000,-0.9107363401,1.0000000000,0.2875419750,1.0000000000,1.0000000000,-0.0475051254,-0.5420144689,0.6529737327,0.4655522262,0.8209913389,0.7559613732,-0.7061084964,1.0000000000,0.1633146279,0.3776109157,0.0172110310,-0.3583067155,1.0000000000,0.2875763103,-0.0431965672,-0.3455128603,1.0000000000,-0.8051528097,0.4047964483,1.0000000000,-0.6033324124,-0.9121960857,1.0000000000,0.2088663588,0.1803472430,-0.5019493544] ,
[-0.2167844191,-1.0000000000,-1.0000000000,1.0000000000,-1.0000000000,0.5806805694,-0.6133401520,0.0244194942,-0.8217725847,-0.9873923508,-1.0000000000,-1.0000000000,0.4617013960,0.8891033761,-1.0000000000,0.9876482684,-0.9223942330,-0.9179337907,1.0000000000,0.3088997766,-0.1172372110,-0.3208293159,-1.0000000000,-0.0189605582,0.8175370323,-0.6656516620,0.8154041346,-0.3632351176,1.0000000000,-0.5773221849,-0.7335357110,-1.0000000000] ,
[-0.3644342547,0.9113814191,-0.1367254087,0.9912588227,0.3895789620,-1.0000000000,0.1154418909,1.0000000000,-0.4434639579,0.2475934223,0.4662058484,-0.5804357714,-0.9158183151,-0.2353560423,-0.2556852669,-0.2311854713,0.7733752754,-1.0000000000,-0.0749506856,0.8179353683,-0.1738234550,-1.0000000000,1.0000000000,0.0108559515,-1.0000000000,-1.0000000000,0.3898024753,0.3685458771,0.3010319671,0.2222408886,-0.6947992480,1.0000000000] 
]