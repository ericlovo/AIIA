import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from 'playwright'
const base = process.env.STUDIO_URL
assert.ok(base, 'Run through npm run test:browser or an isolated Studio preview')
const realStore = process.env.MEETINGS_FIXTURE === '1'
if (realStore) assert.equal(new URL(base).port, '5876')
const out = process.env.SCREENSHOT_DIR || join(tmpdir(), 'aiia-meetings-proof')
await mkdir(out, { recursive: true })
const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH})
try {
  const calls = []
  const failures = []
  async function session(width) {
    const page = await browser.newPage({viewport:{width,height:width===390?844:900}})
    page.setDefaultTimeout(10000)
    page.on('pageerror', e=>failures.push(e.message))
    await page.routeWebSocket('**/ws', ws=>ws.onMessage(()=>{}))
    await page.route('**/api/**', async r => {
      const req=r.request(), path=new URL(req.url()).pathname
      const ok = data => r.fulfill({json:data})
      if(path.startsWith('/api/clients')) {
        if(realStore) return r.continue()
        const [, , , client, , id, action] = path.split('/')
        if(!client) return ok({clients:[{id:'smart-medical',name:'Smart Medical'},{id:'gci',name:'GCI'}]})
        if(!id && req.method()==='GET') return ok({meetings:calls.filter(c=>c.client_id===client)})
        if(!id) {
          const call={...req.postDataJSON(),client_id:client,id:'test-call',notes:[],note_count:0}
          calls.push(call); return ok(call)
        }
        const call=calls.find(c=>c.id===id && c.client_id===client)
        if(!call)return r.fulfill({status:404,json:{detail:'Unknown meeting'}})
        if(action==='notes'){
          const note={...req.postDataJSON(),id:'n'+call.notes.length,created_at:new Date().toISOString()}
          call.notes.push(note);call.note_count++;return ok(note)
        }
        return ok(call)
      }
      const fixtures={
        '/api/agents': {agents:[]}, '/api/assignments':{assignments:[]},'/api/handoffs':{handoffs:[]},
        '/api/git-workspaces':{workspaces:[]},'/api/git-writes':{writes:[]},'/api/agents/resources':{repos:[],github:{status:'disconnected'}},
        '/api/memory-inbox':{ideas:[],total:0},'/api/tasks':[], '/api/health':{},'/api/monitor':{services:{}},
      }
      return ok(fixtures[path] || {})
    })
    return page
  }
  const desktop=await session(1280), phone=await session(390)
  await desktop.goto(base+'#/meetings/smart-medical')
  await desktop.getByRole('button',{name:'New call'}).click()
  await desktop.getByLabel('Call title').fill('Fixture review')
  await desktop.getByLabel(/Start time/).fill('2026-11-12T10:00')
  await desktop.getByRole('button',{name:'Save call'}).click()
  await desktop.getByRole('heading',{name:'Add a note'}).waitFor()
  const link=desktop.url()
  await phone.goto(link)
  await desktop.getByLabel('Contributor').fill('Contributor A')
  await desktop.getByLabel('Source',{exact:true}).selectOption('zoom')
  await desktop.getByLabel('Notes',{exact:true}).fill('The two sample units are ready for review.')
  await desktop.getByRole('button',{name:'Save note'}).click()
  await phone.getByLabel('Contributor').fill('Contributor B')
  await phone.getByLabel('Source',{exact:true}).selectOption('morrow')
  await phone.getByLabel('Load a text file').setInputFiles({name:'morrow-fixture.md', mimeType:'text/markdown',buffer:Buffer.from('Confirm the revised delivery date with the team.')})
  await phone.getByRole('button',{name:'Save note'}).click()
  for(const [page,label] of [[desktop,'desktop'],[phone,'phone']]){
    await page.locator('article').getByText('The two sample units are ready for review.').waitFor()
    await page.locator('article').getByText('Confirm the revised delivery date with the team.').waitFor()
    await page.getByRole('heading',{name:'Fixture review'}).scrollIntoViewIfNeeded()
    assert.equal(await page.locator('main').evaluate(el=>el.scrollWidth>el.clientWidth+1),false)
    await page.screenshot({path:join(out,'shared-call-'+label+'.png')})
  }
  await desktop.getByRole('link',{name:'Create work with these notes'}).click()
  assert.equal(await desktop.getByLabel('Client',{exact:true}).inputValue(),'smart-medical')
  assert.equal(await desktop.getByLabel('Call',{exact:true}).inputValue(),new URLSearchParams(link.split('?')[1]).get('call'))
  await phone.getByRole('link',{name:'GCI',exact:true}).click()
  await phone.getByText('No calls recorded yet.').waitFor()
  assert.deepEqual(failures,[])
  console.log('Two contributors shared one call, imported text, scoped work and isolated the second client')
} finally {await browser.close()}
