export async function openStudioView(page, label) {
  const nav = page.getByRole('navigation', { name: 'Studio', exact: true }).filter({ visible: true })
  const name = label === 'Assignments' ? 'Work' : label
  const target = nav.getByRole('link', { name, exact: true })
  if (!['Today', 'Inbox', 'Jobs', 'Work', 'Projects'].includes(name) && !await target.isVisible()) {
    await nav.locator('summary[aria-label="Studio tools"]').click()
  }
  await target.click()
}
