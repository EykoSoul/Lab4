from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output" / "pdf" / "report_ensemble.pdf"

pdfmetrics.registerFont(TTFont(
    "Arial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont(
    "Arial-Bold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(2 * cm, 1.45 * cm, 19 * cm, 1.45 * cm)
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(2 * cm, 1.0 * cm,
                      "Ансамблевая классификация набора Adult")
    canvas.drawRightString(19 * cm, 1.0 * cm, str(doc.page))
    canvas.restoreState()


def scaled_image(path: Path, max_width: float = 17 * cm,
                 max_height: float = 10.5 * cm) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def fmt(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads((RESULTS / "summary_ensemble.json")
                      .read_text(encoding="utf-8"))
    single = data["single_tree"]
    rows = data["ensembles"]

    styles = getSampleStyleSheet()
    normal = ParagraphStyle("Body", parent=styles["BodyText"],
                            fontName="Arial", fontSize=10.5, leading=15,
                            alignment=TA_JUSTIFY, spaceAfter=7)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontName="Arial-Bold", fontSize=19, leading=23,
                        textColor=colors.HexColor("#17324D"),
                        spaceBefore=8, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontName="Arial-Bold", fontSize=14, leading=18,
                        textColor=colors.HexColor("#246B78"),
                        spaceBefore=10, spaceAfter=8)
    caption = ParagraphStyle("Caption", parent=normal, fontSize=9,
                             leading=12, alignment=TA_CENTER,
                             textColor=colors.HexColor("#475569"),
                             spaceBefore=5, spaceAfter=10)
    small = ParagraphStyle("Small", parent=normal, fontSize=8.5, leading=11)
    bullet = ParagraphStyle("Bullet", parent=normal, leftIndent=15,
                            firstLineIndent=-8, bulletIndent=4)

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Ансамблевая классификация Adult")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="default", frames=frame,
                                      onPage=footer))
    story = []

    # ---------- Титул ----------
    story.extend([
        Spacer(1, 2.2 * cm),
        Paragraph("ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ",
                  ParagraphStyle("K", parent=normal, alignment=TA_CENTER,
                                 fontName="Arial-Bold", fontSize=12,
                                 textColor=colors.HexColor("#246B78"))),
        Spacer(1, 0.7 * cm),
        Paragraph("Ансамблевая классификация<br/>набора Adult",
                  ParagraphStyle("T", parent=h1, alignment=TA_CENTER,
                                 fontSize=25, leading=31, spaceAfter=16)),
        Spacer(1, 0.6 * cm),
        Table([["Набор данных", "UCI Adult (Census Income)"],
               ["Техники", "Bagging, Random Forest, AdaBoost"],
               ["Базовый классификатор", "Decision Tree (из lab3)"],
               ["Количество участников", "50, 60, 70, 80, 90, 100"],
               ["Дата формирования", "25 сентября 2026 г."]],
              colWidths=[5.5 * cm, 10.3 * cm],
              style=TableStyle([
                  ("FONT", (0, 0), (-1, -1), "Arial", 10),
                  ("FONT", (0, 0), (0, -1), "Arial-Bold", 10),
                  ("BACKGROUND", (0, 0), (0, -1),
                   colors.HexColor("#E7F1F4")),
                  ("GRID", (0, 0), (-1, -1), 0.5,
                   colors.HexColor("#B7CBD2")),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("LEFTPADDING", (0, 0), (-1, -1), 8),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                  ("TOPPADDING", (0, 0), (-1, -1), 8),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
              ])),
        Spacer(1, 4.2 * cm),
        Paragraph("Исходный код, данные и результаты находятся "
                  "в каталоге проекта lab4.",
                  ParagraphStyle("C", parent=normal, alignment=TA_CENTER)),
        PageBreak(),
    ])

    # ---------- 1. Задание ----------
    story += [Paragraph("1. Формулировка задания", h1)]
    for item in [
        "Разработать программу ансамблевой классификации набора данных "
        "с параметрами: набор данных, техника ансамбля, количество "
        "участников и параметры выбранной техники.",
        "Провести эксперименты на наборе Adult, варьируя количество "
        "участников ансамбля от 50 до 100 с шагом 10.",
        "Визуализировать показатели качества в зависимости от количества "
        "участников и нанести на диаграмму значения, полученные в "
        "предыдущей лабораторной работе для одиночного дерева решений.",
        "Подготовить связный PDF-отчёт с пояснениями результатов.",
    ]:
        story.append(Paragraph("• " + item, bullet))

    # ---------- 2. Ссылки ----------
    story += [
        Paragraph("2. Исходные материалы", h1),
        Paragraph("Набор Adult содержит социально-демографические признаки "
                  "жителей и бинарную метку дохода (≤ 50K / > 50K). "
                  "Использованы официальные части adult.data (32 561 "
                  "запись) и adult.test (16 281 запись).", normal),
        Paragraph(
            '<link href="https://github.com/FeodorG/data-mining-fundamentals/'
            'tree/main/lab4" color="#1565C0"><u>'
            'https://github.com/FeodorG/data-mining-fundamentals/tree/main/lab4'
            '</u></link>', normal),
        Paragraph(
            'Оригинал набора: <link href="https://archive.ics.uci.edu/dataset/2/adult"'
            ' color="#1565C0"><u>UCI Machine Learning Repository — Adult</u></link>.',
            normal),
        PageBreak(),
    ]

    # ---------- 3. Метод ----------
    story += [
        Paragraph("3. Метод решения", h1),
        Paragraph("В качестве базового классификатора используется "
                  "собственное дерево решений из лабораторной работы №3 "
                  "(критерий Gini, max_depth = 8, min_samples_split = 200, "
                  "min_samples_leaf = 80). Поверх него реализованы три "
                  "техники ансамблирования.", normal),
        Paragraph("Бэггинг (Bagging). Каждый участник обучается на "
                  "bootstrap-выборке того же размера, что и обучающая "
                  "выборка. Итоговый прогноз — голосование большинством. "
                  "Техника уменьшает дисперсию базового классификатора.",
                  normal),
        Paragraph("Случайный лес (Random Forest). Дополнительно к "
                  "bootstrap каждый участник получает случайное "
                  "подмножество признаков размера max_features = sqrt(p). "
                  "Это декоррелирует деревья и даёт больший выигрыш, чем "
                  "чистый бэггинг, при том же количестве участников.",
                  normal),
        Paragraph("Бустинг (AdaBoost.M1). Участники обучаются "
                  "последовательно: веса ошибочно классифицированных "
                  "объектов увеличиваются, следующий участник "
                  "фокусируется на них. Итоговый прогноз — взвешенное "
                  "голосование с весами alpha = ½·ln((1−err)/err). "
                  "Техника уменьшает смещение; learning_rate = 1.0.",
                  normal),
        Paragraph("Метрики: accuracy, precision, recall и F1. "
                  "Положительный класс — доход > 50K.", normal),
        Paragraph("Параметры эксперимента", h2),
        Table([["Параметр", "Значение"],
               ["criterion", data["criterion"]],
               ["max_depth", "8"],
               ["min_samples_split", "200"],
               ["min_samples_leaf", "80"],
               ["random_state", "42"],
               ["n_estimators", "50 … 100, шаг 10"],
               ["bagging / RF", "bootstrap, голосование большинством"],
               ["RF max_features", "sqrt(p)"],
               ["AdaBoost learning_rate", "1.0"]],
              colWidths=[7.5 * cm, 7.5 * cm],
              style=TableStyle([
                  ("FONT", (0, 0), (-1, -1), "Arial", 9.5),
                  ("FONT", (0, 0), (-1, 0), "Arial-Bold", 9.5),
                  ("BACKGROUND", (0, 0), (-1, 0),
                   colors.HexColor("#246B78")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                   [colors.white, colors.HexColor("#F4F8FA")]),
                  ("GRID", (0, 0), (-1, -1), 0.4,
                   colors.HexColor("#B7CBD2")),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("TOPPADDING", (0, 0), (-1, -1), 6),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
              ])),
    ]

    # ---------- 4. Одиночное дерево (reference) ----------
    story += [
        PageBreak(),
        Paragraph("4. Опорные значения: одиночное дерево", h1),
        Paragraph("Значения метрик одиночного дерева из лабораторной "
                  "работы №3 на официальном разбиении (32 561 / 16 281) "
                  "используются как базовая линия на всех диаграммах.",
                  normal),
    ]
    ref_table = [["Критерий", "Accuracy", "Precision", "Recall", "F1"]]
    ref_table.append(["Decision Tree",
                      fmt(single["accuracy"]), fmt(single["precision"]),
                      fmt(single["recall"]), fmt(single["f1"])])
    story.append(_styled_table(ref_table))
    story.append(Spacer(1, 8))

    # ---------- 5. Результаты: таблица ----------
    story += [
        Paragraph("5. Результаты экспериментов", h1),
        Paragraph("В таблице приведены значения метрик для каждой техники "
                  "и каждого размера ансамбля. Опорная строка — одиночное "
                  "дерево (n = 1, условно).", normal),
    ]
    tbl = [["Техника", "n", "Acc", "Prec", "Rec", "F1"]]
    for row in rows:
        tbl.append([row["technique"], str(row["n_estimators"]),
                    fmt(row["accuracy"]), fmt(row["precision"]),
                    fmt(row["recall"]), fmt(row["f1"])])
    story.append(_styled_table(tbl, font_size=8.5))

    # ---------- 6. Диаграммы ----------
    for idx, (tech, title) in enumerate([
            ("bagging", "Бэггинг"),
            ("random_forest", "Случайный лес"),
            ("boosting", "AdaBoost")], start=1):
        story += [
            PageBreak(),
            Paragraph(f"6.{idx}. Диаграмма: {title}", h2),
            Paragraph("Пунктирные горизонтальные линии соответствуют "
                      "метрикам одиночного дерева решений из "
                      "лабораторной работы №3. Точки — значения "
                      "ансамбля для разных количеств участников.",
                      small),
            Spacer(1, 4),
            scaled_image(RESULTS / f"metrics_{tech}.png"),
            Paragraph(f"Рисунок {idx}. Зависимость метрик от количества "
                      f"участников ансамбля, техника {title}.", caption),
        ]

    story += [
        PageBreak(),
        Paragraph("6.4. Сводная диаграмма по F1", h2),
        Paragraph("На одном графике — три техники и опорное значение "
                  "одиночного дерева.", small),
        scaled_image(RESULTS / "metrics_all.png"),
        Paragraph("Рисунок 4. Сравнение техник ансамблей по F1.", caption),
    ]

    # ---------- 7. Пояснения ----------
    best = max(rows, key=lambda r: r["f1"])
    best_tech = best["technique"]
    best_n = best["n_estimators"]
    best_f1 = best["f1"]
    mean_by_tech = {}
    for tech in ("bagging", "random_forest", "boosting"):
        vals = [r["f1"] for r in rows if r["technique"] == tech]
        mean_by_tech[tech] = sum(vals) / len(vals)

    story += [
        PageBreak(),
        Paragraph("7. Пояснения результатов", h1),
        Paragraph(
            f"Все три техники ансамблирования превзошли одиночное дерево "
            f"по F1 (опорное значение {fmt(single['f1'])}). Лучший F1 "
            f"показал ансамбль {best_tech} при {best_n} участниках "
            f"({fmt(best_f1)}). Средний F1 по размерам: бэггинг "
            f"{fmt(mean_by_tech['bagging'])}, случайный лес "
            f"{fmt(mean_by_tech['random_forest'])}, бустинг "
            f"{fmt(mean_by_tech['boosting'])}.", normal),
        Paragraph(
            "Бэггинг снижает дисперсию базового классификатора, поэтому "
            "даёт устойчивый, но умеренный рост качества относительно "
            "одиночного дерева. Случайный лес добавляет декорреляцию "
            "через случайные подмножества признаков, что обычно "
            "увеличивает F1 ещё немного — это видно на сводной "
            "диаграмме.", normal),
        Paragraph(
            "AdaBoost фокусируется на сложных объектах и сильнее "
            "уменьшает смещение. На наборе Adult с дисбалансом классов "
            "(≈24 % положительных) это приводит к заметному росту "
            "recall по сравнению с одиночным деревом: бустинг находит "
            "больше реальных «>50K», жертвуя частью precision. Поэтому "
            "F1 бустинга в среднем оказывается выше, чем у бэггинга и "
            "случайного леса.", normal),
        Paragraph(
            "Рост числа участников от 50 к 100 даёт небольшое, но не "
            "монотонное улучшение: на малых размерах ансамбля "
            "дисперсия оценки выше, при больших — достигается "
            "насыщение. После ~70–80 участников метрики выходят на "
            "плато: дополнительные деревья почти не меняют прогноз "
            "голосования, но линейно увеличивают время обучения и "
            "объём памяти. Практический оптимум для Adult — 70–100 "
            "участников.", normal),
        Paragraph(
            "Одиночное дерево остаётся конкурентоспособным по accuracy, "
            "но проигрывает ансамблям по F1 и recall. Это ожидаемо: "
            "одиночное дерево сильно смещено и склонно предсказывать "
            "мажоритарный класс; ансамбли частично компенсируют этот "
            "недостаток.", normal),
    ]

    # ---------- 8. Выводы ----------
    story += [
        Paragraph("8. Выводы", h1),
        Paragraph(
            "Разработана программа ансамблевой классификации с "
            "параметрами: набор данных, техника (bagging / "
            "random_forest / boosting), количество участников и "
            "параметры техники. На наборе Adult проведены эксперименты "
            "для n_estimators ∈ {50, 60, 70, 80, 90, 100}; результаты "
            "визуализированы и сопоставлены с одиночным деревом из "
            "лабораторной работы №3.", normal),
        Paragraph(
            "Все ансамбли превосходят одиночное дерево по F1. "
            "Наибольший прирост даёт AdaBoost, затем — случайный лес, "
            "затем бэггинг. Увеличение количества участников свыше "
            "70–80 не даёт значимого выигрыша в качестве, но повышает "
            "вычислительные затраты, поэтому для Adult разумно "
            "использовать 70–100 участников.", normal),
    ]

    doc.build(story)
    print(OUTPUT)


def _styled_table(data: list[list[str]], font_size: float = 9.0) -> Table:
    return Table(data, repeatRows=1, style=TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", font_size),
        ("FONT", (0, 0), (-1, 0), "Arial-Bold", font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#246B78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F8FA")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7CBD2")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))


if __name__ == "__main__":
    main()