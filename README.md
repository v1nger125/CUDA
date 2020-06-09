# CUDA
Лаббораторные работы  
Язык C++  

Гайд по запуску/установке на windows 10 или что-то вроде этого  
Быстрый запуск - смотри пункт 4

1. Устанавливаем Visual studio c++ 2019, пользовательской версии будет достаточно(возможно работает и на более ранних версиях) https://visualstudio.microsoft.com/ru/vs/features/cplusplus/
2. Устанавливаем Cuda, если все пойдет как надо, то в конце установки, будет написано, что установлено расширение на vs https://developer.nvidia.com/cuda-downloads
3. Запускаем vs, создаем проект, если все сделанно правильно среди типов проектов, можно будет найти Cuda runtime, vs создаст настроенный проект с простым приложением - сложение двух векторнов.
4. Поскольку я не знаком с особенностями vs(как и с самим C++) и с тем как работать в ней с git, я выгрузил проекты целиком со всеми exe и прочим(а еще мне очень не хотелось прописывать gitignore), для запуска можно скопировать исходный код в изначальный проект cuda(шаги 1-3), его можно найти по следующему пути LABNAME/LABNAME/kernel.cu или выгрузить проекты целиком(они не очень большие) и запустить exe файл по следующему пути LABNAME/x64/debug/LABNAME.exe

Примечание:
Все комментарии на русском полетели, комментарии на английском - остаток от простого проекта, сгенерированного vs, я решил оставить некоторые из них, потому что в целом они довольно полезны и помогли мне быстро вспомнить, как работать с cuda. Информация и русских комментариев продублированна в readme в соответствующих папках.  
В 0 и 2 лаббораторной работе, закомментированный код предназначен для ручной проверки, на малых размерах
