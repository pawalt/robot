"""Real-browser playground checks, executed on Modal."""
from pathlib import Path
import json
import modal
from modal_resources import PROJECT, volume

app = modal.App('robot-browser-check')
image = (modal.Image.debian_slim(python_version='3.14')
         .uv_sync(uv_project_dir=str(PROJECT), groups=['dev'])
         .run_commands('playwright install --with-deps chromium')
         .add_local_file(PROJECT / 'modal_resources.py', '/root/modal_resources.py'))
URL = 'https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-play'

@app.function(image=image, cpu=2, memory=2048, timeout=600, volumes={'/data': volume})
def check():
    from playwright.sync_api import sync_playwright
    results = {}
    screenshots = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox'])
        for name, width, height, mobile in [('desktop',1440,1000,False),('mobile',390,844,True),('small-mobile',320,700,True)]:
            context = browser.new_context(viewport={'width':width,'height':height}, is_mobile=mobile, has_touch=mobile, device_scale_factor=1)
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            response = page.goto(URL, wait_until='networkidle')
            assert response.status == 200
            page.wait_for_function("!document.getElementById('run').disabled")
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), name+' overflows'
            page.locator('#run').click()
            page.wait_for_function("Number(document.getElementById('time').textContent)>0.5")
            page.locator('#run').click()
            assert page.locator('#status').inner_text() == 'Paused'
            page.locator('#reset').click()
            assert page.locator('#time').inner_text() == '0.0'
            page.locator('[data-preset=bottom]').click()
            page.locator('#upper').fill('30')
            page.locator('#upper').dispatch_event('input')
            assert page.locator('#upperValue').inner_text() == '30°'
            page.locator('#motor').uncheck()
            page.locator('#run').click()
            assert page.locator('#status').inner_text() == 'Motor off'
            page.locator('#motor').check()
            page.locator('#kick').click()
            assert 'KICK' in page.locator('#event').inner_text()
            page.locator('#run').click()
            page.locator('[data-preset=tilt]').click()
            page.locator('#canvas').scroll_into_view_if_needed()
            box = page.locator('#canvas').bounding_box()
            x, y = box['x']+box['width']/2, box['y']+box['height']/2
            if mobile:
                session = context.new_cdp_session(page)
                session.send('Input.dispatchTouchEvent', {'type':'touchStart','touchPoints':[{'x':x,'y':y-60}]})
                session.send('Input.dispatchTouchEvent', {'type':'touchMove','touchPoints':[{'x':x+50,'y':y-30}]})
                session.send('Input.dispatchTouchEvent', {'type':'touchEnd','touchPoints':[]})
            else:
                page.mouse.move(x,y-60)
                page.mouse.down()
                page.mouse.move(x+50,y-30,steps=5)
                page.mouse.up()
            assert page.locator('#upperValue').inner_text() != '15°', 'Drag did not change pose'
            page.locator('[data-preset=tilt]').click()
            page.evaluate('scrollTo(0,0)')
            # Exercise both commanded directions and the zero-torque display.
            for command, expected_left in [(1, '50%'), (-1, '0%'), (0, '50%')]:
                page.evaluate('(u) => { torque=u; controls(); draw(); }', command)
                assert page.locator('#effort').evaluate('(e) => e.style.left') == expected_left
                assert page.locator('#effort').evaluate('(e) => e.style.width') == ('0%' if command == 0 else '50%')
            page.evaluate('torque=.75; controls(); draw()')
            screenshots[name] = page.screenshot(full_page=True)
            assert not errors, errors
            results[name] = {'passed':True,'viewport':[width,height],'touch':mobile,'errors':errors}
            context.close()
        browser.close()
    output = Path('/data/browser-verification')
    output.mkdir(parents=True,exist_ok=True)
    (output/'ui.json').write_text(json.dumps(results,indent=2))
    for name, data in screenshots.items():
        (output/f'{name}.png').write_bytes(data)
    volume.commit()
    return results, screenshots

@app.local_entrypoint()
def main():
    results, screenshots = check.remote()
    print(json.dumps(results,indent=2))
    for name, data in screenshots.items():
        Path(f'/tmp/robot-playground-{name}.png').write_bytes(data)
