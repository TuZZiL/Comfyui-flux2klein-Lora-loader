# Бриф для OPUS: Brainstorm наступних Companion conditioning node

Аудиторія: senior engineer / product-minded implementer  
Use context: internal R&D для наступного покоління companion conditioning nodes у FLUX.2 Klein  
Пріоритет: практична користь для редагування зображень  
Definition of done: один markdown-документ з ранжованими ідеями, практичним обгрунтуванням, аналізом перетинів і рекомендованим порядком розробки  
Avoid: код, загальна теорія diffusion, розмиті ідеї, дублікати node, і все, що не може реально допомогти в image editing  
Mode: calm, direct, specific

## Мета

Згенеруй наступну хвилю ідей для `Companion conditioning` node у цьому репозиторії.

Нам потрібен не теоретичний список, а матеріал, який допоможе реально вирішити, що будувати далі для image-editing workflow.

Сконцентруйся на ідеях, які покращують:

- локальний контроль над reference influence
- баланс між prompt і reference
- регіональне редагування
- контроль color drift
- збереження identity
- очистку або заміну background
- стабільність композиції
- конфлікти між кількома reference

## Поточна база

Уже існують такі companion nodes:

- `Flux2KleinRefLatentController`
- `Flux2KleinTextRefBalance`
- `Flux2KleinMaskRefController`
- `Flux2KleinColorAnchor`

Використовуй їх як поточну surface area. Не пропонуй їх повторно під новими назвами, якщо це не є реальною суттєвою еволюцією.

## Що вже роблять існуючі node

Вважай, що така поведінка вже реалізована:

- `Ref Latent Controller`
  - змінює, наскільки сильно reference tokens залишаються в attention path
  - може зміщувати coarse appearance проти fine detail
  - підтримує optional spatial fade і таргетинг одного або всіх reference
- `Text/Ref Balance`
  - зміщує контроль між prompt і reference
  - `attn_patch` - м'який default
  - `latent_mix` - сильніше втручання
- `Mask Ref Controller`
  - застосовує spatial mask до reference latent
  - `scale` послаблює reference influence
  - `mix` замінює masked regions іншим latent signal
  - є inversion маски
- `Color Anchor`
  - робить color correction під час sampling
  - тримає output ближче до source palette
  - може враховувати variance каналів

## Дослідницьке питання

Думай як практичний користувач image editor, а не як researcher diffusion.

Відповідь на запитання:

що реально зменшить фрустрацію під час редагування?

Шукай node, які вирішують такі задачі:

- зберегти face, але змінити outfit
- змінити background, не руйнуючи identity
- підсилити prompt тільки в певній зоні
- не дати hands, hair або logos поплисти
- зберегти structure, але дозволити texture changes
- вирівняти palette по всьому edit
- керувати конфліктом кількох reference
- стабілізувати composition без повного заморожування зображення

## Напрями, які варто розглянути

Використовуй це як seed-напрямки, але не обмежуйся ними:

1. Region Prompt Router
- маршрутизація різної prompt-ваги в різні masked regions
- корисно, коли один prompt має керувати background, а інший - subject

2. Subject / Background Split Controller
- окремий контроль reference influence для subject і background
- корисно для portrait swap, зміни environment і product shot

3. Face / Hair / Hands Protector
- спеціалізований захист найбільш проблемних зон
- може бути wrapper над існуючими primitives або окремий node

4. Edge-Aware Reference Fade
- пом'якшення різких переходів на межах маски
- корисно, коли маска чиста, але шов все одно видно

5. Multi-Reference Arbiter
- розв'язання конфліктів між кількома reference
- корисно для комбінацій style + identity + product reference

6. Local Color Spill Shield
- зберігає palette в одній зоні, дозволяючи іншій змінюватися
- корисно, коли color drift локалізований, а не глобальний

7. Composition Anchor
- утримує horizon, framing або загальну layout-структуру
- корисно для background replacement і scene edits

8. Step-Sensitive Regional Bias
- змінює регіональну силу залежно від denoising stage
- корисно, коли ранні steps мають тримати structure, а пізні - давати freedom

9. Semantic Mask Helper
- робить маски кориснішими через region presets або mask transforms
- корисно, коли користувач знає цільову область, але не точну геометрію

10. Prompt Pressure Splitter
- робить prompt сильнішим в одній зоні та слабшим в іншій
- корисно, коли reference правильний глобально, але потрібна локальна зміна

## Що пріоритезувати

Надавай перевагу ідеям, які відповідають більшості цих критеріїв:

- пояснюються одним реченням
- вирішують реальний pain у редагуванні
- добре лягають на існуючу model / conditioning / mask архітектуру
- компонується з наявними node
- не потребують важких зовнішніх залежностей
- достатньо відрізняються, щоб виправдати окремий node

## Чого уникати

Не пропонуй:

- просто перейменовані версії існуючих чотирьох node
- великі підсистеми, без яких node не буде корисною
- зовнішні segmentation або detection моделі, якщо це не чітко optional
- ідеї, які не можна нормально використати в графі
- вузькі edge case, які не допомагають поширеним workflow

## Формат виходу

Поверни один markdown-документ з такими секціями:

1. Executive summary
2. Ранжований список node-ідей
3. Для кожної ідеї:
   - proposed node name
   - яку проблему вирішує
   - що змінює в graph
   - inputs / outputs
   - default starting values
   - як користувач її тюнить
   - ймовірні failure modes
   - overlap з існуючими node
   - build complexity
   - recommended priority
4. Top 3 node для наступної розробки
5. Ідеї, які варто відкласти або відкинути
6. Shared primitives, які можна повторно використати в коді
7. Naming або UX notes

## Рубрика оцінки

Оцінюй кожну ідею за такими факторами:

- user value
- implementation simplicity
- composability
- overlap risk
- edit-workflow relevance

Якщо дві ідеї перетинаються, скажи це прямо і рекомендуй один із варіантів:

- об'єднати їх в один node
- розширити існуючий node
- залишити один як wrapper для UX

## Сильна перевага

Якщо вагаєшся, обирай те, що найчастіше допомагає звичайному користувачу image editing:

- portraits
- clothing edits
- background replacement
- product image cleanup
- identity preservation
- composition cleanup

## Фінальна вимога

Не давай нам generic brainstorm.
Дай decision-oriented документ, який ми зможемо використати для вибору наступних companion conditioning node з упевненістю.
