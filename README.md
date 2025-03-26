# Сomposites
Training ML and NN models for composites params prediction.

Обучение моделей классического машинного обучения и нейронных сетей для прогнозирования параметров композиционных материалов.
Прилагается тетрадь Jupyter Notebook с кодом обучения и код для запуска приложения для прогнозирования параметров.

--------------------------------------------------------------------------------
Пример датасета:
|Индекс | Соотношение матрица-наполнитель	| Плотность, кг/м3 |	Модуль упругости, ГПа	| Количество отвердителя, м.%	| Содержание эпоксидных групп,%_2	| Температура вспышки, С_2	| Поверхностная плотность, г/м2 |	Модуль упругости при растяжении, ГПа	| Прочность при растяжении, МПа	| Потребление смолы, г/м2	| Угол нашивки, град |	Шаг нашивки	| Плотность нашивки|
|----|-------------|------------|-------------|-------------|-----------|-------------|-------------|-----------|-------------|--------------|----|-----------|-----------|
|374 |	3.950501	| 1910.632051	| 700.907095	| 152.821434	| 22.333254	| 235.940152	| 143.030854	| 68.830496	| 2434.857214	| 265.837278	| 0	  | 5.907932	| 55.948455|
|784	| 2.815082	| 1886.864085	| 1157.430119	| 113.336615	| 19.444788	| 346.604427	| 109.292979	| 76.918789	| 2717.988811	| 275.680811	| 90	| 4.331443	| 71.673397|
|380	| 1.988732	| 2008.300311	| 866.596263	| 82.558627 	| 26.647004	| 274.563206	| 342.540268	| 78.202731	| 2407.873386	| 235.171423	| 0	  | 8.465672	| 50.636439|
|313	| 4.107086	| 1989.461473	| 678.579157	| 98.431047 	| 20.888454	| 294.194381	| 444.498714	| 71.949778	| 1944.712515	| 263.523445	| 0  	| 7.401052	| 66.918583|
|404	| 3.563509	| 1861.493546	| 815.993641	| 134.855988	| 23.340665	| 227.332974	| 153.853595	| 79.916185	| 3089.128812	| 343.515980	| 0	  | 3.570410	| 64.040965|
--------------------------------------------------------------------------------

#### Содержание репозитория:

1) Jupyter Notebook c основным кодом исследования ``composites.ipynb`` 
2) Материалы приложения
   
а) ``app_pytorch.py`` код запуска приложения с моделью pytorch

б) ``app_tf.py`` код запуска приложения с моделью tensorflow

в) ``templates/main.html`` код для html-страницы приложения

г) ``dataset`` набор данных композиционных материалов

д) ``torch_model.pt`` модель pytorch

е) ``tf_model.tf`` модель tensorflow
