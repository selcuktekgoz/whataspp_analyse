import numpy as np
import pandas as pd
import seaborn as sns
from os import path
import re
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as sw
from warnings import filterwarnings
from datetime import datetime
import locale
from tabulate import tabulate

locale.setlocale(locale.LC_TIME, "tr_TR.UTF-8")


filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.1f" % x)
pd.set_option("display.width", 500)

# Giriş ve çıkış dosya isimleri
input_file = "datasets/wp_data.txt"
output_file = "datasets/wp_data_output.txt"

# Silinmek istenen kelimeler
keywords = [
    "kişisini ekledi",
    "kişisini çıkardı",
    "olarak değiştirdi",
    "medya dahil edilmedi",
    "silindi",
    "ayrıldı",
    "gruba eklendi",
    "olan grup adını",
    "açıklamasını değiştirdi",
    "grubunu oluşturdu",
    "whatsapp da dahil",
    "sizi ekledi",
    "bu grubun ayarlarını",
    "davet bağlantısıyla katıldı",
    "grubun simgesini değiştirdi",
    "(dosya ekli)",
    "mesaj bekleniyor",
    "artık yöneticisiniz",
    "bir mesajı sabitledi",
    "bu mesajı sildiniz",
]


def veri_onisleme(kullanici_adim):
    telefon_pattern = r";(\d{1,3} \d{3} \d{3} \d{2} \d{2}):"
    tarih_saat_pattern = r"\b\d{1,2}\.\d{1,2}\.\d{4} \d{2}:\d{2}\b"
    new_lines = []

    tarih_saat_pattern = r"\b\d{1,2}\.\d{1,2}\.\d{4} \d{2}:\d{2}\b"

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.lower() for line in lines]
        lines = [
            line for line in lines if not any(keyword in line for keyword in keywords)
        ]
        for line in lines:
            match = re.search(tarih_saat_pattern, line)
            if match:
                # Desen varsa, satırı yeni_lines listesine ekle
                new_lines.append(line.strip())
            else:
                # Desen yoksa, önceki satıra birleştir
                if new_lines:
                    new_lines[-1] += " " + line.strip()

        new_lines = [line.replace(";", " ") for line in new_lines]
        new_lines = [line.replace(" - ‎~ ", ";") for line in new_lines]
        new_lines = [line.replace(" - +9", ";") for line in new_lines]
        new_lines = [
            re.sub(
                telefon_pattern, lambda match: match.group(0).replace(":", ";"), line
            )
            for line in new_lines
        ]
        new_lines = [
            line.replace(f" - {kullanici_adim}:", f";{kullanici_adim};")
            for line in new_lines
        ]

    with open(output_file, "w", encoding="utf-8") as file:
        for line in new_lines:
            file.write(line)
            file.write("\n")


veri_onisleme("selçuk tekgöz")


def gun_ismi(tarih_str):
    tarih_obj = datetime.strptime(tarih_str, "%d.%m.%Y")
    return tarih_obj.strftime("%A")


def ay_ismi(tarih_str):
    tarih_obj = datetime.strptime(tarih_str, "%d.%m.%Y")
    return tarih_obj.strftime("%B")


def mesai_durumu(saati, gunu):
    if gunu == "Cumartesi" or gunu == "Pazar":
        return "Mesai dışı"
    elif (gunu != "Cumartesi" and gunu != "Pazar") and (
        saati >= "08:00" and saati <= "18:00"
    ):
        return "Mesai içi"
    else:
        return "Mesai dışı"


def mesaj_zamani(saati, gunu):
    if gunu == "Cumartesi" or gunu == "Pazar":
        return "Mesai dışı"
    elif saati >= "08:00" and saati <= "13:00":
        return "Öğleden önce"
    elif saati > "13:00" and saati <= "18:00":
        return "Öğleden sonra"
    else:
        return "Mesai dışı"


def df_cevir(output_file):
    df = pd.read_csv(output_file, delimiter=";", header=None)
    df.columns = ["TARIH_SAAT", "GONDEREN", "MESAJ"]
    df["TARIH"] = df["TARIH_SAAT"].map(lambda x: x.split(" ")[0])
    df["SAAT"] = df["TARIH_SAAT"].map(lambda x: x.split(" ")[1])
    df["AY"] = df["TARIH"].map(ay_ismi)
    df["GUN"] = df["TARIH"].map(gun_ismi)

    gun_sirasi = [
        "Pazartesi",
        "Salı",
        "Çarşamba",
        "Perşembe",
        "Cuma",
        "Cumartesi",
        "Pazar",
    ]
    ay_sirasi = [
        "Ocak",
        "Şubat",
        "Mart",
        "Nisan",
        "Mayıs",
        "Haziran",
        "Temmuz",
        "Ağustos",
        "Eylül",
        "Ekim",
        "Kasım",
        "Aralık",
    ]
    df["AY"] = pd.Categorical(df["AY"], categories=ay_sirasi, ordered=True)
    df["GUN"] = pd.Categorical(df["GUN"], categories=gun_sirasi, ordered=True)
    df["MESAI_SAATI_MI"] = list(map(mesai_durumu, df["SAAT"], df["GUN"]))
    df["MESAJ_ARALIGI"] = list(map(mesaj_zamani, df["SAAT"], df["GUN"]))
    df.drop("TARIH_SAAT", axis=1, inplace=True)
    # df.to_excel("deneme.xlsx")
    return df


df = df_cevir(output_file)


class CloudFromDoc(WordCloud):
    def __init__(
        self,
        file_path="doc.txt",
        add_stopwords=["ilgili", "olarak"],
        background_color="white",
        width=1500,
        height=1000,
        color=322,
        maxwords=1000,
        horizontal_ratio=0.75,
        collocation_threshold=30,
        hue=571,
        saturation=None,
        lightness=None,
        output=None,
    ):
        self.path = file_path
        self.stopwords = sw.words()
        self.stopwords.extend(add_stopwords)
        self.width = width
        self.height = height
        self.color = color
        self.maxwords = maxwords
        self.horizontal = horizontal_ratio
        self.collocation_thresh = collocation_threshold
        self.bg_color = background_color
        self.text = self._read_document()
        self.hue = hue
        self.saturation = saturation
        self.lightness = lightness
        self.output = output

        print(f"Wordcloud {self.path} oluşturuluyor...")
        print(f"Wordcloud {self.width} x {self.height} pixel olarak oluşturuldu..")
        print(
            f'Wordcloud özellikleri: hue-{"random" if self.hue == None else str(self.hue)}, saturation-{"random" if self.saturation == None else str(self.saturation)}, lightness-{"random" if self.lightness == None else str(self.ligthness)}'
        )

        self.cloud = WordCloud(
            background_color=self.bg_color,
            width=self.width,
            height=self.height,
            stopwords=self.stopwords,
            max_words=self.maxwords,
            prefer_horizontal=self.horizontal,
            collocation_threshold=self.collocation_thresh,
        )

        self.cloud.generate(self.text)

        # show
        plt.figure(figsize=[50, 30])
        plt.imshow(
            self.cloud.recolor(color_func=self.custom_color_func), interpolation="sinc"
        )
        plt.axis("off")

        # # store to file
        self.save_wc()

        plt.show()

    def _read_document(self):
        with open(self.path, "r", encoding="utf-8") as myfile:
            data = myfile.readlines()
        text = ",".join(data)
        return text

    def custom_color_func(self, **kwargs):
        return f"hsl({np.random.randint(0, 360) if self.hue == None else self.hue}, {np.random.randint(15, 100) if self.saturation == None else self.saturation}%, {np.random.randint(0, 60) if self.lightness == None else self.lightness}%)"

    def save_wc(self):
        hue = "" if self.hue == None else "H" + str(self.hue)
        saturation = "" if self.saturation == None else "S" + str(self.saturation)
        lightness = "" if self.lightness == None else "L" + str(self.lightness)
        if self.output == None:
            output = f"wc_Size{self.width}_{self.height}_hslColor{'Random' if (self.hue == None and self.saturation == None and self.lightness == None) else f'{hue}{saturation}{lightness}'}.png"
        self.cloud.to_file(output)
        print(f"Wordcloud {output} olarak kaydedildi..")


CloudFromDoc(file_path="datasets/wp_data_output.txt", hue=221, saturation=41)


def cross_tab(dataframe, col1, col2, siralama=None):
    print(f"\n{col1} X {col2} Frekans Dağılımı")
    cross_table = pd.crosstab(
        dataframe[col1], dataframe[col2], margins=True, margins_name="Toplam"
    )
    tablo = tabulate(cross_table, headers="keys", tablefmt="pretty")
    print(tablo)
    cross_table.to_excel(f"{col1}x{col2}.xlsx")


cross_tab(df, "AY", "GUN")
cross_tab(df, "GUN", "MESAI_SAATI_MI")
cross_tab(df, "GUN", "MESAJ_ARALIGI")


def ilk_n(n):
    ilk_n_kisi = df["GONDEREN"].value_counts().nlargest(n).sort_values(ascending=False)
    tablo = tabulate(
        ilk_n_kisi.reset_index().rename(
            columns={"GONDEREN": "GONDEREN", "count": "MESAJ SAYISI"}
        ),
        headers="keys",
        showindex=False,
        tablefmt="pretty",
    )
    print(tablo)

    ilk_n_kisi_df = ilk_n_kisi.reset_index()
    ilk_n_kisi_df.columns = ["GONDEREN", "MESAJ SAYISI"]
    ilk_n_kisi_df.to_excel(f"ilk{n}_kisi.xlsx", index=False)

    return ilk_n_kisi_df


ilk_10 = ilk_n(10)

############# grafikler ####################


custom_palette_1 = {"Mesai dışı": "#ff4a4a", "Mesai içi": "#50c878"}
custom_palette_2 = {
    "Mesai dışı": "#ff4a4a",
    "Öğleden önce": "#ddf6ee",
    "Öğleden sonra": "#50c878",
}


def histogram_ciz(
    dataframe,
    x,
    w=16,
    h=8,
    bins="auto",
    y_ekseni=None,
    ylim=None,
    hue=None,
    palet=None,
    title=None,
    xrotation=None,
):
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(w, h))
    sns.despine(f)
    sns.histplot(
        dataframe,
        x=x,
        hue=hue,
        multiple="stack",
        palette=palet,
        edgecolor=".3",
        linewidth=0.5,
        bins=bins,
    )
    plt.title(title)
    plt.ylabel(y_ekseni)
    plt.xlabel("")
    if xrotation is not None:
        plt.xticks(rotation=xrotation)
    plt.grid(False)
    plt.ylim(0, ylim)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.plot()


histogram_ciz(
    df,
    x="GUN",
    y_ekseni="Mesaj Sayısı",
    ylim=100,
    hue="MESAI_SAATI_MI",
    palet=custom_palette_1,
    title="MESAJLARIN GÜNLERE VE MESAİ DURUMUNA GÖRE DAĞILIMI",
)

histogram_ciz(
    df,
    x="GUN",
    y_ekseni="Mesaj Sayısı",
    ylim=100,
    hue="MESAJ_ARALIGI",
    palet=custom_palette_2,
    title="MESAJLARIN GÜNLERE VE MESAİ ARALIĞINA GÖRE DAĞILIMI",
)

histogram_ciz(
    df,
    x="AY",
    w=8,
    h=4,
    y_ekseni="Mesaj Sayısı",
    ylim=200,
    hue="MESAJ_ARALIGI",
    palet=custom_palette_2,
    title="MESAJLARIN AYLARA VE MESAİ ARALIĞINA GÖRE DAĞILIMI",
)

histogram_ciz(
    df,
    x="AY",
    w=8,
    h=4,
    y_ekseni="Mesaj Sayısı",
    ylim=200,
    hue="MESAI_SAATI_MI",
    palet=custom_palette_1,
    title="MESAJLARIN AYLARA VE MESAİ DURUMUNA GÖRE DAĞILIMI",
)

histogram_ciz(
    df,
    x="TARIH",
    y_ekseni="Mesaj Sayısı",
    hue="MESAI_SAATI_MI",
    palet=custom_palette_1,
    title="MESAJLARIN TARİHE VE MESAİ DURUMUNA GÖRE DAĞILIMI",
    xrotation=90,
    ylim=70,
    w=10,
    h=5,
)

histogram_ciz(
    df,
    x="TARIH",
    y_ekseni="Mesaj Sayısı",
    title="MESAJLARIN TARİHE GÖRE DAĞILIMI",
    xrotation=90,
    ylim=70,
    w=10,
    h=5,
)


def barplot(df, x, y, w=16, h=8, title=None, y_ekseni=None, ylim=None, xrotation=None):
    f, ax = plt.subplots(figsize=(w, h))
    sns.despine(f)
    sns.barplot(
        df,
        x=x,
        y=y,
        edgecolor=".3",
        linewidth=0.5,
    )
    plt.title(title)
    plt.ylabel(y_ekseni)
    plt.xlabel("")
    if xrotation is not None:
        plt.xticks(rotation=xrotation)
    plt.grid(False)
    plt.ylim(0, ylim)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.plot()


barplot(
    ilk_10,
    "GONDEREN",
    "MESAJ SAYISI",
    title="MESAJ GÖNDERİM SAYILARI (İLK 10 KİŞİ)",
    y_ekseni="Mesaj Sayısı",
    ylim=50,
    xrotation=45,
)
