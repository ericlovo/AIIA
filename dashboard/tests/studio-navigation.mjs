export async function openStudioView(page, label) {
  let nav = page.getByRole('navigation', { name: 'Studio', exact: true }).filter({ visible: true })
  if (!await nav.count()) {
    const details = page.getByRole('button', { name: 'Details', exact: true }).first()
    if (await details.count()) await details.click()
    else await page.evaluate(() => { window.location.hash = '#/today' })
    await page.getByRole('navigation', { name: 'Studio', exact: true }).waitFor()
    nav = page.getByRole('navigation', { name: 'Studio', exact: true }).filter({ visible: true })
  }
  const name = label === 'Assignments' ? 'Work' : label
  const target = nav.getByRole('link', { name, exact: true })
  if (!['Today', 'Inbox', 'Jobs', 'Work', 'Projects'].includes(name) && !await target.isVisible()) {
    await nav.locator('summary[aria-label="Studio tools"]').click()
  }
  await target.click()
}
