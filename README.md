# Anomaly Detection Tool


Dieses Repository enthält das entwicklete Tool von Marsel Perdoci im Rahmen der Masterarbeit mit dem Titel "Entwicklung eines Tools zur Erkennung von Anomalien in Serverdaten mittels Machine Learning".

## Online-Zugriff

Das Tool ist auch als Webanwendung verfügbar und kann online über die Streamlit Community Cloud aufgerufen werden. Sie können darauf zugreifen, indem Sie [diesen Link](https://mp-adt.streamlit.app/) besuchen.

## Datensätze

Im Ordner `datasets` sind Datensätze von [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) zum zur Verwendung im Tool zu finden.
Die Datensätze beinhalten:
1. unbeschriftete Daten (Ordner `realAWSCloudwatch`)
2. beschriftete Daten mit Anomalien (Ordner `realAWSCloudwatch - labeled`)
3. beschriftete Daten ohne Anomalien (Ordner `realAWSCloudwatch - labeled - no anomaly`)

Die Beschriftungen aus 2. und 3. wurden programmatisch aus dem [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) Framework extrahiert.

Weitere Datensätze finden Sie unter https://github.com/numenta/NAB/tree/master/data

## Voraussetzungen für die lokale Bereitstellung des Tools

Bevor Sie beginnen, stellen Sie sicher, dass Sie die folgenden Anforderungen erfüllen:

* Sie haben einen `Windows`-Rechner mit Python 3.10.
* Sie haben die neueste Version von pip installiert. Wenn nicht, können Sie es [hier](https://pip.pypa.io/en/stable/installation/) installieren.

## Installation des Tools

Um das Tool zu installieren, folgen Sie diesen Schritten:

1. Klonen Sie das Repository auf Ihren lokalen Rechner:

```bash
git clone https://github.com/Perdoci/mp_adt.git
```

2. Navigieren Sie zum geklonten Repository:

```bash
cd mp_adt
```

## Installation von Abhängigkeiten

Dieses Projekt hat einige Abhängigkeiten, die Sie mit pip installieren können. Die Abhängigkeiten sind in der Datei `requirements.txt` aufgeführt. Um diese Abhängigkeiten zu installieren, führen Sie den folgenden Befehl aus:

```bash
pip install -r requirements.txt
```

## Ausführen der Anwendung

Um die Streamlit-Anwendung lokal auszuführen, verwenden Sie den folgenden Befehl:

```bash
streamlit run home.py
```

Dies startet den Streamlit-Server und die Anwendung ist in Ihrem Webbrowser unter `localhost:8501` zugänglich.


## Lokale Bereitstellung

Die obigen Schritte führen Sie durch das Ausführen der Anwendung lokal auf Ihrem Rechner. Wenn Sie die Anwendung auf einem Server oder in einer anderen Umgebung bereitstellen möchten, müssen Sie möglicherweise zusätzliche Schritte je nach Ihrer Bereitstellungsumgebung befolgen.

## Kontakt

Wenn Sie mich kontaktieren möchten, erreichen Sie mich unter `marsel.perdoci(at)hotmail(punkt)com`.

## Lizenz

Dieses Projekt verwendet die folgende Lizenz: `Creative Commons Zero v1.0 Universal`.
