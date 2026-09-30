export async function openStudioView(page, label) {
  const nav = page.getByRole('navigation', { name: 'Studio', exact: true }).filter({ visible: true })
  const mobile = await nav.getByText('Work', { exact: true }).count() > 0
  if (mobile && !['Today', 'Agents', 'Assignments'].includes(label)) {
    const target = nav.getByRole('link', { name: label, exact: true })
    if (!await target.isVisible()) await nav.getByText('Studio', { exact: true }).click()
  }
  await nav.getByRole('link', { name: mobile && label === 'Assignments' ? 'Work' : label, exact: true }).click()
}
