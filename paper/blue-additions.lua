function Span(el)
  if el.classes:includes('addition') and FORMAT:match('latex') then
    table.insert(el.content, 1, pandoc.RawInline('latex', '\\textcolor{blue}{'))
    table.insert(el.content, pandoc.RawInline('latex', '}'))
    return el.content
  end
end

function Table(el)
  if #el.colspecs == 5 and pandoc.utils.stringify(el.head):find('Fixed Python certified', 1, true) then
    el.colspecs = {{pandoc.AlignLeft, 0.24}, {pandoc.AlignRight, 0.16},
                  {pandoc.AlignRight, 0.20}, {pandoc.AlignRight, 0.20},
                  {pandoc.AlignRight, 0.20}}
    if FORMAT:match('latex') then
      return {pandoc.RawBlock('latex', '\\begingroup\\small\\setstretch{1}'),
              el, pandoc.RawBlock('latex', '\\endgroup')}
    end
  end
  if #el.colspecs == 2 and pandoc.utils.stringify(el.head):find('Instruction') then
    el.colspecs = {{pandoc.AlignRight, 0.06}, {pandoc.AlignLeft, 0.94}}
  end
  if #el.colspecs == 6 and pandoc.utils.stringify(el.head):find('Qualified') and pandoc.utils.stringify(el.head):find('Path-Alg', 1, true) then
    el.colspecs = {{pandoc.AlignRight, 0.07}, {pandoc.AlignRight, 0.07},
                  {pandoc.AlignRight, 0.14}, {pandoc.AlignRight, 0.22},
                  {pandoc.AlignRight, 0.19}, {pandoc.AlignLeft, 0.31}}
    if FORMAT:match('latex') then
      return {pandoc.RawBlock('latex', '\\begingroup\\small\\setstretch{1}'),
              el, pandoc.RawBlock('latex', '\\endgroup')}
    end
  end
  if #el.colspecs == 6 and pandoc.utils.stringify(el.head):find('Fixture', 1, true) then
    el.colspecs = {{pandoc.AlignLeft, 0.24}, {pandoc.AlignLeft, 0.12},
                  {pandoc.AlignRight, 0.13}, {pandoc.AlignRight, 0.12},
                  {pandoc.AlignRight, 0.20}, {pandoc.AlignLeft, 0.19}}
    if FORMAT:match('latex') then
      return {pandoc.RawBlock('latex', '\\begingroup\\small\\setstretch{1}'),
              el, pandoc.RawBlock('latex', '\\endgroup')}
    end
  end
  if #el.colspecs == 5 and pandoc.utils.stringify(el.head):find('Fixed CPU', 1, true) then
    el.colspecs = {{pandoc.AlignLeft, 0.36}, {pandoc.AlignRight, 0.16},
                  {pandoc.AlignRight, 0.16}, {pandoc.AlignRight, 0.16},
                  {pandoc.AlignLeft, 0.16}}
    if FORMAT:match('latex') then
      return {pandoc.RawBlock('latex', '\\begingroup\\small\\setstretch{1}'),
              el, pandoc.RawBlock('latex', '\\endgroup')}
    end
  end
  return el
end

function Image(el)
  if FORMAT:match('html') then
    el.src = el.src:gsub('%.pdf$', '.png')
    return el
  end
end
