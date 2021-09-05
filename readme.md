
# Faktoranalyse (Faktor Analyse)

Bei Faktoranalyse geht es um die Untersuchung von Daten und Datasets, um darin gewisse Ursachen zu finden, warum Daten sich auf eine bestimmte Art und Weise verhalten, warum sie in einer bestimmten Art und Weise wirken.
Das Wichtige daran ist, dass es sich um sogenannte latente Variablen, Faktoren handelt. Das sind Variablen, die aussagekräftig sind, aber nicht direkt beobachtbar, sondern abgeleitet aus anderen Fakten.


Beispiel:
Sie wollen eine Segmentierung von Kunden vornehmen, das heißt, Kunden in verschiedene Gruppen einteilen. Sie haben nun Daten aus einer Kundenbefragung und können diese Faktoranalyse anwenden als einfaches Mittel, um die Befragten in aussagekräftige Segmente zu gruppieren.
Diese basiert zum Beispiel auf Ähnlichkeiten in der Art und Weise, wie die Befragten bestimmte Teilmenge von dieser Umfrage beantwortet haben.


Beispiel 1:
Sie wollen eine Segmentierung von Kunden vornehmen, das heißt, Kunden in verschiedene Gruppen einteilen. Sie haben nun Daten aus einer Kundenbefragung und können diese Faktoranalyse anwenden als einfaches Mittel, um die Befragten in aussagekräftige Segmente zu gruppieren.
Diese basiert zum Beispiel auf Ähnlichkeiten in der Art und Weise, wie die Befragten bestimmte Teilmenge von dieser Umfrage beantwortet haben.

Um eine Faktoranalyse ausführen zu können, gibt es gewisse Annahmen, die mit den Daten getroffen werden.

#1-Die Features müssen metrisch sein
#2-Die Funktionen sind kontinuierlich oder ordinal
#3- Es gibt eine sogenannte r-Korrelation, größer als 0,3 zwischen den Features und ihrem Dataset. 
#4-100 Beobachtungen insgesamt und pro Merkmal mehr als 5 Beobachtungswerte 
#5- Die Probe muss homogen sein, das heißt, gleichmäßig verteilt.


Beispiel 2:
Um dieses Beispiel durchzuführen werden wir mit dem iris-Datenquelle arbeiten. Er ist standardmäßig bei sklearn integriert.


Der Iris-Datensatz enthält vier numerische Variablen, die drei verschiedene Arten von Iris-Blüten beschreiben. Das sind hier alles numerische Attribute, was man ja fordert bei der Faktoranalyse.
                       
Iris Blumen			Attribute (Vorhersagemermale-prective features)
•Setosa             •Sepal lenght
•Versicolor         •Sepal width
•Virgnica           •Pedal lenght
                    •Pedal width
wir werden jetzt versuchen eine Kombination von Features aufzudecken, die die meisten Informationen, die meisten Aussagen enthalten. Das werden dann diese latenten Variablen sein, von denen wir gerade reden


```python
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import FactorAnalysis
from sklearn  import datasets

```


```python
iris=datasets.load_iris()
X=iris.data
variable_names=iris.feature_names
X[0:10,]



```




    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2],
           [5.4, 3.9, 1.7, 0.4],
           [4.6, 3.4, 1.4, 0.3],
           [5. , 3.4, 1.5, 0.2],
           [4.4, 2.9, 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.1]])



Faktoranalyse auf dem Iris-Dataset


```python
factor= FactorAnalysis().fit(X)
pd.DataFrame(factor.components_ ,columns=variable_names)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.707227</td>
      <td>-0.153147</td>
      <td>1.653151</td>
      <td>0.701569</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.114676</td>
      <td>0.159763</td>
      <td>-0.045604</td>
      <td>-0.014052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Interpretation der Ergebnisse

Wir haben also folgende Daten ermittelt bezüglich der Länge und Breite von Kelch und Blütenblätter. 
Dabei fallen zwei Faktoren mit Zahlen auf, die von 0 abweichen. Dieser Faktor 1 scheint sehr stark sich gegenüber dem Faktor 2 herauszukristallisieren, die Werte sind viel höher. 


Offensichtlich hat dieser Faktor 1 einen großen Einfluss auf die Kelchblattlänge, die Blütenblattlänge und auch die Blütenblattbreite von Iris-Pflanzen. Faktor 2 hat keine große Auswirkung auf eine dieser Variablen. Nun geht es bei der Faktoranalyse darum die relevanten Daten zu betrachten, die irrelevanten aber wegzulassen, das bedeutet, man hat eine sogenannte Redaktion der Dimension. 

Und dadurch, dass Faktor 2 ganz offensichtlich keine große Auswirkung auf die betrachteten Eigenschaften hat, kann man diese Variable eigentlich auch weglassen, ohne dass viel Auswirkung daraus entsteht. Man kann also Faktor 2 im Laufe der weiteren Bewertung einfach streichen und wir haben in dieser Analyse eine zugrunde liegende latente Variable isoliert, Faktor 1, die auf die Kelchblattlänge, die Blütenblattlänge und Blütenblattbreite großen Einfluss hat.
